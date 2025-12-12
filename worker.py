#!/usr/bin/env python3
"""
worker.py - ADST 2.0 demo worker (auto loop for MAX_EPOCHS)
- local training on data/site{ID}
- pairwise mask exchange (direct TCP)
- send encrypted masked gradients over UDP per epoch
- listen for rotation packets on TCP and update K_EPOCH
"""
import argparse, os, json, socket, struct, threading, time, secrets, hashlib
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

MAX_EPOCHS = 10

from model import SmallCNN

KEY_DIR = "keys"; LOG_DIR = "logs"
os.makedirs(KEY_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

def log_event(evt):
    evt['ts'] = datetime.now(timezone.utc).isoformat()
    fname = os.path.join(LOG_DIR, f"adst_worker_{evt['worker']}.log")
    with open(fname, "a") as f: f.write(json.dumps(evt) + "\n")

def atomic_write(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp,"wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def hkdf_derive(shared: bytes, info=b"adst transport key", length=32):
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(shared)



# ---------- identity (Ed25519) ----------
def get_or_create_ed25519(worker_id):
    fn = os.path.join(KEY_DIR, f"worker_{worker_id}_ed.key")
    if os.path.exists(fn):
        b = open(fn, "rb").read()
        return ed25519.Ed25519PrivateKey.from_private_bytes(b)
    else:
        priv = ed25519.Ed25519PrivateKey.generate()
        atomic_write(fn, priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        ))
        log_event({"component":"worker","event":"gen_ed25519","worker":worker_id})
        return priv

# ---------- handshake ----------
def do_handshake(coord_host, coord_port, worker_id):
    ed_priv = get_or_create_ed25519(worker_id)
    ed_pub = ed_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.connect((coord_host, coord_port))
    hello = {'worker_id': worker_id, 'ed25519_pub_hex': ed_pub.hex()}; sock.sendall(json.dumps(hello).encode())
    b = sock.recv(65536)
    j = json.loads(b.decode())
    coord_x_pub = bytes.fromhex(j['coord_x_pub_hex']); coord_x_sig = bytes.fromhex(j['coord_x_sig_hex']); coord_ed_pub = bytes.fromhex(j['coord_ed_pub_hex'])
    ed25519.Ed25519PublicKey.from_public_bytes(coord_ed_pub).verify(coord_x_sig, coord_x_pub)
    w_x_priv = x25519.X25519PrivateKey.generate(); w_x_pub = w_x_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    w_sig = ed_priv.sign(w_x_pub); sock.sendall(json.dumps({'worker_x_pub_hex': w_x_pub.hex(), 'worker_x_sig_hex': w_sig.hex()}).encode())
    shared = w_x_priv.exchange(x25519.X25519PublicKey.from_public_bytes(coord_x_pub))
    transport_key = hkdf_derive(shared, info=b"adst transport key")
    # receive epoch message
    b2 = sock.recv(65536)
    epoch_num = struct.unpack(">I", b2[:4])[0]
    enc_len = struct.unpack(">H", b2[4:6])[0]; enc_blob = b2[6:6+enc_len]
    nonce = enc_blob[:12]; ct = enc_blob[12:]
    aes = AESGCM(transport_key); k_epoch = aes.decrypt(nonce, ct, None)
    log_event({"component":"worker","event":"handshake_complete","worker":worker_id,"epoch":epoch_num})
    return sock, transport_key, epoch_num, k_epoch, w_x_priv

# ---------- mask listener & peer send ----------
mask_shares_incoming = {}
def start_mask_listener(listen_ip, listen_port, worker_id, x_priv):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind((listen_ip, listen_port)); s.listen(8)
    def runner():
        while True:
            conn, addr = s.accept()
            try:
                data = conn.recv(65536); j = json.loads(data.decode())
                frm = int(j['from']); peer_x_pub = bytes.fromhex(j['x_pub_hex'])
                nonce = bytes.fromhex(j['nonce_hex']); enc = bytes.fromhex(j['enc_hex'])
                shared = x_priv.exchange(x25519.X25519PublicKey.from_public_bytes(peer_x_pub))
                K = hkdf_derive(shared, info=b"adst pairwise mask")
                aes = AESGCM(K); aad = b"mask-share"
                pt = aes.decrypt(nonce, enc, aad)
                arr = np.frombuffer(pt, dtype=np.float32); mask_shares_incoming[frm] = arr
                log_event({"component":"worker","event":"mask_received","worker":worker_id,"from":frm})
            except Exception as e:
                log_event({"component":"worker","event":"mask_receive_error","worker":worker_id,"err":str(e)})
            finally:
                try: conn.close()
                except: pass
    t = threading.Thread(target=runner, daemon=True); t.start(); return s

def send_mask_to_peer(peer_ip, peer_port, peer_x_pub_hex, x_priv, my_id, peer_id, vec):
    try:
        peer_x_pub = bytes.fromhex(peer_x_pub_hex)
        shared = x_priv.exchange(x25519.X25519PublicKey.from_public_bytes(peer_x_pub))
        K = hkdf_derive(shared, info=b"adst pairwise mask"); aes = AESGCM(K)
        nonce = secrets.token_bytes(12); aad = b"mask-share"; ct = aes.encrypt(nonce, vec.tobytes(), aad)
        frame = {'from': my_id, 'x_pub_hex': x_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw).hex(), 'nonce_hex':nonce.hex(), 'enc_hex': ct.hex()}
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(2.0)
        s.connect((peer_ip, peer_port)); s.sendall(json.dumps(frame).encode()); s.close()
        log_event({"component":"worker","event":"sent_mask","worker":my_id,"to":peer_id})
    except Exception as e:
        log_event({"component":"worker","event":"mask_send_failed","worker":my_id,"to":peer_id,"err":str(e)})

# ---------- local training ----------
def local_train_and_get_grads(data_dir):
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    model = SmallCNN(); model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in loader:
        opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb); loss.backward(); opt.step()
        break
    grads = []
    for p in model.parameters():
        grads.append(p.grad.view(-1).cpu().numpy())
    flat_grads = np.concatenate(grads)

    # --- Differential Privacy (DP) ---
    # 1. Gradient Clipping
    DP_CLIP_NORM = 1.0
    total_norm = np.linalg.norm(flat_grads)
    if total_norm > DP_CLIP_NORM:
        flat_grads = flat_grads * (DP_CLIP_NORM / total_norm)
    
    # 2. Gaussian Noise
    DP_NOISE_SCALE = 0.01
    noise = np.random.normal(0, DP_NOISE_SCALE, flat_grads.shape)
    return flat_grads + noise

# ---------- send gradient over UDP ----------
def make_aad(jobid, epoch, sender, seq, chunk, total, dtype):
    return struct.pack(">Q I I I I H", jobid, epoch, sender, seq, chunk, total) + struct.pack(">H", dtype)

def send_gradient_udp(udp_host, udp_port, jobid, epoch, K_epoch, worker_id, seq, dtype, grad_array, tcp_conn):
    payload = json.dumps({'grads': grad_array.tolist()}).encode()
    CHUNK_SIZE = 12000
    chunks = [payload[i:i+CHUNK_SIZE] for i in range(0, len(payload), CHUNK_SIZE)]
    total = len(chunks); aes = AESGCM(K_epoch); framed={}
    for i,ch in enumerate(chunks):
        nonce = secrets.token_bytes(12); aad = make_aad(jobid, epoch, worker_id, seq, i, total, dtype)
        ct = aes.encrypt(nonce, ch, aad)
        framed[i] = {'nonce':nonce.hex(), 'ct':ct.hex()}
        send_udp_chunk(udp_host, udp_port, jobid, epoch, worker_id, seq, i, total, dtype, nonce, ct)
        time.sleep(0.002)
    # wait for possible NACKs for a short window
    tcp_conn.settimeout(3.0)
    try:
        data = tcp_conn.recv(65536)
        if data:
            j = json.loads(data.decode())
            if j.get('type') == 'nack' and int(j.get('seq')) == seq:
                missing = j.get('missing', [])
                for m in missing:
                    info = framed.get(m)
                    if info:
                        nonce = bytes.fromhex(info['nonce']); ct = bytes.fromhex(info['ct'])
                        send_udp_chunk(udp_host, udp_port, jobid, epoch, worker_id, seq, m, total, dtype, nonce, ct)
                log_event({"component":"worker","event":"nack_handled","worker":worker_id,"seq":seq,"missing":missing})
    except Exception:
        pass
    tcp_conn.settimeout(None)

def send_udp_chunk(udp_host, udp_port, jobid, epoch, sender, seq, chunk, total, dtype, nonce, ct):
    j = {'jobid': str(jobid),'epoch': str(epoch),'sender': str(sender),'seq': str(seq),'chunk': str(chunk),'total': str(total),'dtype': str(dtype),'nonce_hex': nonce.hex(),'payload_hex': ct.hex()}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(json.dumps(j).encode(), (udp_host, udp_port)); s.close()

# ---------- main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--coord-host", default="127.0.0.1")
    p.add_argument("--coord-port", type=int, default=9000)
    p.add_argument("--udp-port", type=int, default=9001)
    p.add_argument("--jobid", type=lambda x: int(x,0), default=0x1122334455667788)
    args = p.parse_args()

    worker_id = args.id
    data_dir = os.path.join("data", f"site{worker_id}")
    if not os.path.isdir(data_dir):
        print(f"[!] Data folder {data_dir} not found. Please add local data."); exit(1)

    tcp_conn, K_T, epoch_num, K_EPOCH, x_priv = do_handshake(args.coord_host, args.coord_port, worker_id)
    # start mask listener on ephemeral port and announce
    listen_ip = "127.0.0.1"
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); lsock.bind((listen_ip,0)); listen_port = lsock.getsockname()[1]; lsock.close()
    start_mask_listener(listen_ip, listen_port, worker_id, x_priv)
    tcp_conn.sendall(json.dumps({'type':'announce_listener','ip':listen_ip,'port':listen_port}).encode())
    time.sleep(0.3)
    tcp_conn.sendall(json.dumps({'type':'get_peers'}).encode())
    peers = []
    try:
        b = tcp_conn.recv(65536); j = json.loads(b.decode()); peers = j.get('peers', [])
    except Exception:
        pass

    # pairwise masks: everyone with id>me sends a random share; everyone with id<me will have sent to us
    vector_len = 0  # will set after computing grads first epoch
    current_epoch = epoch_num
    seq_counter = 0

    while current_epoch <= MAX_EPOCHS:
        # local training -> produce grads (flat)
        grads = local_train_and_get_grads(data_dir)
        if vector_len == 0:
            vector_len = grads.size
        # create pairwise shares for peers with id > me
        outgoing = {}
        for p in peers:
            pid = int(p['id'])
            if pid == worker_id: continue
            if pid > worker_id:
                vec = (np.random.randn(vector_len).astype(np.float32) * 0.01)
                outgoing[pid] = (p['ip'], int(p['port']), p['x_pub_hex'], vec)
        # send outgoing shares
        for pid, (ip_, port_, x_pub_hex, vec) in outgoing.items():
            send_mask_to_peer(ip_, port_, x_pub_hex, x_priv, worker_id, pid, vec)
        # small wait to collect incoming shares
        time.sleep(1.0)
        # build mask: sum(incoming) - sum(outgoing)
        mask = np.zeros(vector_len, dtype=np.float32)
        for k,v in mask_shares_incoming.items(): mask += v
        for pid, (_,_,_,vec) in outgoing.items(): mask -= vec
        masked = grads + mask
        # send masked gradient over UDP
        seq = int(time.time() * 1000) & 0xffffffff; seq_counter += 1
        send_gradient_udp(args.coord_host, args.udp_port, args.jobid, current_epoch, K_EPOCH, worker_id, seq, 0x01, masked, tcp_conn)
        log_event({"component":"worker","event":"udp_chunk_sent","worker":worker_id,"seq":seq})
        # wait for rotation packet from coordinator (blocking)
        # coordinator will send pack_epoch_message(new_epoch, enc_blob, key_hash) encrypted with current K_EPOCH
        try:
            b = tcp_conn.recv(65536)
            if not b:
                break
            # parse rotation packet
            new_epoch = struct.unpack(">I", b[:4])[0]
            enc_len = struct.unpack(">H", b[4:6])[0]
            enc_blob = b[6:6+enc_len]
            hlen = b[6+enc_len]
            key_hash = b[7+enc_len:7+enc_len+hlen]
            # decrypt enc_blob with current K_EPOCH
            nonce = enc_blob[:12]; ct = enc_blob[12:]
            aes = AESGCM(K_EPOCH)
            try:
                k_next = aes.decrypt(nonce, ct, None)
                # verify key hash optionally (skip in demo)
                K_EPOCH = k_next
                current_epoch = new_epoch
                log_event({"component":"worker","event":"epoch_rotated","worker":worker_id,"epoch":current_epoch})
                # loop continues to train next epoch
            except Exception as e:
                log_event({"component":"worker","event":"rotation_decrypt_failed","worker":worker_id,"err":str(e)})
                break
        except Exception:
            # timeout or connection closed -> exit
            break

    # final keepalive briefly then exit
    time.sleep(0.5)
    try: tcp_conn.close()
    except: pass
    log_event({"component":"worker","event":"done","worker":worker_id})
