import os, time, threading, pickle, queue, copy, random
import flax
import numpy as np

class AsyncSaver:
    def __init__(self, save_dir, params_lock, rl_buffer_lock=None, demo_buffer_lock=None, max_pending=2):
        self.save_dir = save_dir
        self.params_lock = params_lock
        self.rl_buffer_lock = rl_buffer_lock
        self.demo_buffer_lock = demo_buffer_lock
        self.q = queue.Queue(maxsize=max_pending)
        self._alive = True
        self._worker = threading.Thread(target=self._loop, name="async_saver", daemon=True)
        self._worker.start()

    def request(self, *, agent, step, params_ref=None,
                rl_buffer=None, demo_buffer=None,
                log_step=None, rpc_call_count=None, version_id=None,
                flags_snapshot=None, online_rng=None):
        meta = dict(
            rl_buffer=rl_buffer,
            demo_buffer=demo_buffer,
            log_step=int(log_step) if log_step is not None else None,
            rpc_call_count=int(rpc_call_count) if rpc_call_count is not None else None,
            version_id=int(version_id) if isinstance(version_id, (int, np.integer)) else (
                int(version_id.get("val", 0)) if isinstance(version_id, dict) else None
            ),
            flags_snapshot=copy.deepcopy(flags_snapshot) if flags_snapshot is not None else None,
            online_rng = (np.array(online_rng, copy=True) if online_rng is not None else None)
        )
        item = ("full", int(step), agent, params_ref, meta)
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait(); self.q.task_done()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def _snapshot(self, agent, step, params_ref, meta):
        with self.params_lock:
            if params_ref is not None and 'params' in params_ref:
                net = agent.network.replace(params=params_ref['params'])
                agent_to_save = agent.replace(network=net)
            else:
                agent_to_save = agent
        save = dict(agent=flax.serialization.to_state_dict(agent_to_save))

        try:   np_rng = np.random.get_state()
        except Exception: np_rng = None
        try:   py_rng = random.getstate()
        except Exception: py_rng = None
        save["rng"] = dict(jax=agent_to_save.rng, numpy=np_rng, python=py_rng)

        rb = meta.get("rl_buffer"); db = meta.get("demo_buffer")
        if rb is not None:
            if self.rl_buffer_lock is not None:
                with self.rl_buffer_lock:
                    save["rl_buffer"] = rb.state_dict() if hasattr(rb, "state_dict") else self._fallback_buffer_state(rb)
            else:
                save["rl_buffer"] = rb.state_dict() if hasattr(rb, "state_dict") else self._fallback_buffer_state(rb)
        if db is not None:
            if self.demo_buffer_lock is not None:
                with self.demo_buffer_lock:
                    save["demo_buffer"] = db.state_dict() if hasattr(db, "state_dict") else self._fallback_buffer_state(db)
            else:
                save["demo_buffer"] = db.state_dict() if hasattr(db, "state_dict") else self._fallback_buffer_state(db)

        save["meta"] = dict(
            log_step=meta.get("log_step"),
            rpc_call_count=meta.get("rpc_call_count"),
            version_id=meta.get("version_id"),
            flags=meta.get("flags_snapshot"),
            online_rng=meta.get("online_rng"),
            save_time=time.time(),
        )

        self.save(save, f'checkpoint_{step}.pkl')

    def _loop(self):
        last_done = None
        while self._alive:
            try:
                kind, step, agent, params_ref, meta = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if step != last_done:
                    self._snapshot(agent, step, params_ref, meta or {})
                    last_done = step
            finally:
                self.q.task_done()

    def flush_and_stop(self, *, agent=None, step=None, params_ref=None,
                       rl_buffer=None, demo_buffer=None,
                       log_step=None, rpc_call_count=None, version_id=None,
                       flags_snapshot=None, online_rng=None, timeout=3.0):
        if agent is not None and step is not None:
            self.request(agent=agent, step=step, params_ref=params_ref,
                         rl_buffer=rl_buffer, demo_buffer=demo_buffer,
                         log_step=log_step, rpc_call_count=rpc_call_count,
                         version_id=version_id, flags_snapshot=flags_snapshot, 
                         online_rng=online_rng)
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        self._alive = False

    def _fallback_buffer_state(self, buf):
        out = {}
        for key in ("storage", "size", "capacity", "idx"):
            if hasattr(buf, key):
                val = getattr(buf, key)
                try:
                    import numpy as _np
                    if isinstance(val, (list, tuple, dict)):
                        out[key] = val  # 희망컨대 픽클 가능
                    else:
                        out[key] = _np.array(val, copy=True)
                except Exception:
                    out[key] = val
        return out

    def save(self, obj, filename):
        tmp = os.path.join(self.save_dir, filename + '.tmp')
        final = os.path.join(self.save_dir, filename)
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)