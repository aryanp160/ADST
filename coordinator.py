#!/usr/bin/env python3
"""
coordinator.py - ADST 2.0 demo coordinator (auto 3 epochs)
- TCP control: handshake, peer discovery, rotate epoch keys
- UDP receiver: receive encrypted chunked masked gradients, reassemble, aggregate
- Auto-run: runs exactly MAX_EPOCHS and exits cleanly
"""
import argparse, socket, threading, struct, os, json, hashlib, time, secrets, logging, sys
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

# -----------------
MAX_EPOCHS = 3
# -----------------

# SmallCNN (same shape as workers)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8,8))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------- config -----------------
KEY_DIR = "keys"; LOG_DIR = "logs"
os.makedirs(KEY_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "adst_coord.log")
logger = logging.getLogger("adst_coord")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE); fh.setFormatter(logging.Formatter('%(message)s')); logger.addHandler(fh)
def log_event(evt: dict):
    evt['ts'] = datetime.now(timezone.utc).isoformat(); fh.stream.write(json.dumps(evt) + "\n"); fh.flush()

# runtime
workers = {}         # worker_id -> info dict
workers_lock = threading.Lock()
JOB_ID = None
EPOCH = 0
K_EPOCH = None
K_PREV = None

# reassembly & aggregation
reassembly = {}
reass_lock = threading.Lock()
REASSEMBLY_TIMEOUT = 1.0
agg_state = {}  # agg_state[epoch][worker] = np.array(flat grads)
global_model = None
val_transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

# ----------------- crypto helpers -----------------
def atomic_write(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    try: os.chmod(path, 0o600)
    except: pass

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def hkdf_derive(shared: bytes, info=b"adst transport key", length=32):
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(shared)

def pack_epoch_message(epoch_num: int, enc_blob: bytes, key_hash: bytes):
    return struct.pack(">I H", epoch_num, len(enc_blob)) + enc_blob + struct.pack(">B", len(key_hash)) + key_hash

# ----------------- key management -----------------
COORD_ED_FILE = os.path.join(KEY_DIR, "coord_ed25519.key")
def get_or_create_coord_ed25519():
    if os.path.exists(COORD_ED_FILE):
        b = open(COORD_ED_FILE, "rb").read()
        return ed25519.Ed25519PrivateKey.from_private_bytes(b)
    else:
        priv = ed25519.Ed25519PrivateKey.generate()
        atomic_write(COORD_ED_FILE, priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        ))
        log_event({"component":"coordinator","event":"gen_coord_ed25519"})
        return priv

def save_epoch_key(jobid: int, epoch: int, key: bytes):
    dpath = os.path.join(KEY_DIR, f"job_{jobid:x}"); os.makedirs(dpath, exist_ok=True)
    fname = os.path.join(dpath, f"epoch_{epoch}.key"); atomic_write(fname, key)
    entries = sorted([f for f in os.listdir(dpath) if f.startswith("epoch_")])
    if len(entries) > 2:
        for rm in entries[:-2]:
            try: os.remove(os.path.join(dpath, rm))
            except: pass

# ----------------- TCP control -----------------
def handle_worker_tcp(conn, addr):
    global EPOCH, K_EPOCH, K_PREV
    try:
        b = conn.recv(65536)
        if not b: conn.close(); return
        hello = json.loads(b.decode()); wid = int(hello['worker_id']); worker_ed_pub = bytes.fromhex(hello['ed25519_pub_hex'])
        with workers_lock:
            workers[wid] = workers.get(wid, {}); workers[wid].update({'tcp_conn': conn, 'tcp_addr': addr, 'ed25519_pub': worker_ed_pub})
        log_event({"component":"coordinator","event":"worker_hello","worker":wid,"addr":str(addr)})

        coord_ed_priv = get_or_create_coord_ed25519()
        coord_ed_pub = coord_ed_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        coord_x_priv = x25519.X25519PrivateKey.generate()
        coord_x_pub_bytes = coord_x_priv.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        sig = coord_ed_priv.sign(coord_x_pub_bytes)
        conn.sendall(json.dumps({'coord_x_pub_hex': coord_x_pub_bytes.hex(), 'coord_x_sig_hex': sig.hex(), 'coord_ed_pub_hex': coord_ed_pub.hex()}).encode())

        b2 = conn.recv(65536)
        wmsg = json.loads(b2.decode()); worker_x_pub = bytes.fromhex(wmsg['worker_x_pub_hex']); worker_x_sig = bytes.fromhex(wmsg['worker_x_sig_hex'])
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(worker_ed_pub).verify(worker_x_sig, worker_x_pub)
        except Exception as e:
            log_event({"component":"coordinator","event":"worker_signature_failed","worker":wid,"err":str(e)}); conn.close(); return

        shared = coord_x_priv.exchange(x25519.X25519PublicKey.from_public_bytes(worker_x_pub))
        transport_key = hkdf_derive(shared, info=b"adst transport key")
        with workers_lock:
            workers[wid].update({'transport_key': transport_key, 'x25519_pub': worker_x_pub})
        log_event({"component":"coordinator","event":"transport_key_derived","worker":wid})

        # send current epoch key encrypted under transport_key
        if K_EPOCH is None:
            EPOCH = 1; K_EPOCH = AESGCM.generate_key(bit_length=256); save_epoch_key(JOB_ID, EPOCH, K_EPOCH)
        aes = AESGCM(transport_key); nonce = secrets.token_bytes(12); enc = aes.encrypt(nonce, K_EPOCH, None)
        key_hash = sha256(struct.pack(">Q I", JOB_ID, EPOCH) + K_EPOCH)
        epoch_msg = pack_epoch_message(EPOCH, nonce + enc, key_hash); conn.sendall(epoch_msg)
        log_event({"component":"coordinator","event":"epoch_sent","worker":wid,"epoch":EPOCH,"key_hash":key_hash.hex()})

        conn.settimeout(0.5)
        while True:
            try:
                data = conn.recv(65536)
                if not data:
                    break
                j = json.loads(data.decode()); typ = j.get('type')
                if typ == 'get_peers':
                    with workers_lock:
                        peers = []
                        for wid2, info in workers.items():
                            if 'listen_ip' in info and 'listen_port' in info and 'x25519_pub' in info:
                                peers.append({'id': wid2, 'ip': info['listen_ip'], 'port': info['listen_port'], 'x_pub_hex': info['x25519_pub'].hex()})
                    conn.sendall(json.dumps({'peers': peers}).encode())
                elif typ == 'announce_listener':
                    with workers_lock:
                        workers[wid].update({'listen_ip': j['ip'], 'listen_port': int(j['port'])})
                    log_event({"component":"coordinator","event":"announce_listener","worker":wid,"ip":j['ip'],"port":j['port']})
                # else ignore other requests (workers wait for rotation)
            except socket.timeout:
                continue
            except Exception:
                break
    except Exception as ex:
        log_event({"component":"coordinator","event":"tcp_handler_error","err":str(ex)})
    finally:
        try: conn.close()
        except: pass
        with workers_lock:
            for wid2, info in list(workers.items()):
                if info.get('tcp_conn') is conn:
                    log_event({"component":"coordinator","event":"worker_disconnected","worker":wid2})
                    info.pop('tcp_conn', None)
                    break

def tcp_server(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind((host, port)); s.listen(32)
    print(f"[C] TCP control listening on {host}:{port}")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle_worker_tcp, args=(conn, addr), daemon=True).start()

# ----------------- UDP receive & aggregate -----------------
def make_aad(jobid, epoch, sender, seq, chunk, total, dtype):
    return struct.pack(">Q I I I I H", jobid, epoch, sender, seq, chunk, total) + struct.pack(">H", dtype)

def get_model_weights(model):
    weights = []
    for p in model.parameters():
        weights.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(weights)

def udp_receiver(host, port, stop_event):
    global global_model, EPOCH, K_EPOCH, K_PREV
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp.bind((host, port))
    print(f"[C] UDP data listening on {host}:{port}")
    while not stop_event.is_set():
        try:
            data, addr = udp.recvfrom(131072)
        except socket.timeout:
            continue
        try:
            j = json.loads(data.decode())
            jobid = int(j['jobid']); epoch = int(j['epoch']); sender = int(j['sender']); seq = int(j['seq'])
            chunk = int(j['chunk']); total = int(j['total']); dtype = int(j['dtype'])
            nonce = bytes.fromhex(j['nonce_hex']); payload = bytes.fromhex(j['payload_hex'])
        except Exception:
            continue
        if epoch != EPOCH:
            log_event({"component":"coordinator","event":"udp_epoch_mismatch","sender":sender,"pkt_epoch":epoch,"current_epoch":EPOCH})
            continue
        try:
            aad = make_aad(jobid, epoch, sender, seq, chunk, total, dtype); aes = AESGCM(K_EPOCH)
            pt = aes.decrypt(nonce, payload, aad)
        except Exception as e:
            log_event({"component":"coordinator","event":"decrypt_failed","sender":sender,"err":str(e)}); continue
        key = (sender, seq)
        with reass_lock:
            state = reassembly.setdefault(key, {'total': total, 'received': {}, 'first_ts': time.time()})
            state['received'][chunk] = pt
        log_event({"component":"coordinator","event":"chunk_received","sender":sender,"seq":seq,"chunk":chunk})
        # if complete -> reassemble and aggregate
        with reass_lock:
            if len(state['received']) == state['total']:
                payloads = [state['received'][i] for i in range(state['total'])]
                full = b"".join(payloads)
                try:
                    msg = json.loads(full.decode())
                    grads_list = msg['grads']; log_event({"component":"coordinator","event":"masked_gradient_received","sender":sender,"seq":seq,"len":len(grads_list)})
                    with workers_lock:
                        agg_state.setdefault(EPOCH, {})[sender] = np.array(grads_list, dtype=np.float32)
                        expected = [w for w, info in workers.items() if 'listen_ip' in info]
                        if len(expected) > 0 and len(agg_state[EPOCH]) >= len(expected):
                            grads = list(agg_state[EPOCH].values()); agg_flat = np.mean(grads, axis=0)
                            if global_model is None: global_model = SmallCNN()
                            idx = 0
                            with torch.no_grad():
                                for p in global_model.parameters():
                                    s = list(p.size()); count = int(np.prod(s))
                                    arr = agg_flat[idx:idx+count].reshape(s)
                                    p -= torch.from_numpy(arr).float() * 0.1
                                    idx += count
                            mean_val = float(np.mean(agg_flat))
                            grad_norm = float(np.linalg.norm(agg_flat))
                            log_event({"component":"coordinator","event":"aggregated_gradient","epoch":EPOCH,"num_workers":len(expected),"mean_gradient":mean_val,"grad_norm":grad_norm})
                            # save model weights
                            global_weights = get_model_weights(global_model)
                            np.save("trained_model.npy", global_weights)
                            # validation
                            val_acc = evaluate_validation(global_model)
                            if val_acc is not None:
                                log_event({"component":"coordinator","event":"validation_accuracy","epoch":EPOCH,"acc":val_acc})
                            # clear for next epoch and trigger rotation
                            agg_state[EPOCH].clear()
                            threading.Thread(target=rotate_epoch_and_broadcast, daemon=True).start()
                except Exception as e:
                    log_event({"component":"coordinator","event":"reassembly_error","err":str(e)})
                reassembly.pop(key, None)

def reassembly_watcher(stop_event):
    while not stop_event.is_set():
        now = time.time(); to_nack=[]
        with reass_lock:
            for key, st in list(reassembly.items()):
                if now - st['first_ts'] > REASSEMBLY_TIMEOUT:
                    sender, seq = key
                    missing = [i for i in range(st['total']) if i not in st['received']]
                    to_nack.append((sender, seq, missing)); st['first_ts'] = now
        for sender, seq, missing in to_nack:
            with workers_lock:
                info = workers.get(sender)
                if info and info.get('tcp_conn'):
                    try:
                        msg = {'type':'nack','seq':seq,'missing':missing}; info['tcp_conn'].sendall(json.dumps(msg).encode())
                        log_event({"component":"coordinator","event":"sent_nack","to":sender,"seq":seq,"missing_count":len(missing)})
                    except Exception as e:
                        log_event({"component":"coordinator","event":"nack_send_failed","to":sender,"err":str(e)})
        time.sleep(0.2)

def evaluate_validation(model):
    val_dir = os.path.join("data", "val")
    if not os.path.isdir(val_dir): return None
    ds = datasets.ImageFolder(val_dir, transform=val_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
    correct = 0; total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb); preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item(); total += xb.size(0)
    model.train()
    return None if total == 0 else correct / total

# ----------------- rotation logic -----------------
def rotate_epoch_and_broadcast():
    global EPOCH, K_EPOCH, K_PREV
    # if we've reached MAX_EPOCHS, shutdown after a moment
    if EPOCH >= MAX_EPOCHS:
        log_event({"component":"coordinator","event":"max_epochs_reached","epoch":EPOCH})
        # let main exit gracefully
        return
    time.sleep(0.5)  # small pause to ensure all worker TCP connections are ready
    new_epoch = EPOCH + 1
    new_key = AESGCM.generate_key(bit_length=256)
    aes_e = AESGCM(K_EPOCH); nonce2 = secrets.token_bytes(12); enc_next = aes_e.encrypt(nonce2, new_key, None)
    key_hash2 = sha256(struct.pack(">Q I", JOB_ID, new_epoch) + new_key)
    rotation_packet = pack_epoch_message(new_epoch, nonce2 + enc_next, key_hash2)
    with workers_lock:
        for wid, info in workers.items():
            tc = info.get('tcp_conn')
            if tc:
                try: tc.sendall(rotation_packet)
                except Exception as e:
                    log_event({"component":"coordinator","event":"rotation_send_failed","to":wid,"err":str(e)})
    K_PREV = K_EPOCH; EPOCH = new_epoch; K_EPOCH = new_key; save_epoch_key(JOB_ID, EPOCH, K_EPOCH)
    log_event({"component":"coordinator","event":"epoch_rotate","epoch":EPOCH,"key_hash":key_hash2.hex()})
    # if reached MAX_EPOCHS, schedule stopper
    if EPOCH >= MAX_EPOCHS:
        log_event({"component":"coordinator","event":"demo_complete","epoch":EPOCH})

# ----------------- main -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--tcp-port", type=int, default=9000)
    p.add_argument("--udp-port", type=int, default=9001); p.add_argument("--jobid", type=lambda x: int(x,0), default=0x1122334455667788)
    args = p.parse_args(); JOB_ID = args.jobid
    # start servers & watcher
    stop_event = threading.Event()
    threading.Thread(target=tcp_server, args=(args.host, args.tcp_port), daemon=True).start()
    udp_thread = threading.Thread(target=udp_receiver, args=(args.host, args.udp_port, stop_event), daemon=True); udp_thread.start()
    threading.Thread(target=reassembly_watcher, args=(stop_event,), daemon=True).start()
    print("[C] Coordinator running. Waiting for workers to join...")

    # wait for enough workers to announce listeners
    while True:
        with workers_lock:
            ready = [w for w,info in workers.items() if 'listen_ip' in info]
        if len(ready) >= 4:
            log_event({"component":"coordinator","event":"all_workers_ready","workers":ready})
            break
        time.sleep(0.5)

    print("[C] All workers ready. Starting auto-run (3 epochs).")
    # Start: send simple 'start' marker by waiting — actual workflow: workers will act immediately after handshake
    # Wait for demo completion
    # Wait until epoch reaches MAX_EPOCHS and last aggregation completed
    while True:
        time.sleep(0.5)
        if EPOCH >= MAX_EPOCHS:
            # give small grace for final logging to flush
            time.sleep(1.0)
            print("[C] Demo finished. Check logs/adst_coord.log and trained_model.npy")
            # stop threads by setting stop_event
            stop_event.set()
            # exit coordinator (process stays for demo manual inspect)
            sys.exit(0)
