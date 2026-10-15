"""Bounded forwarding of unchanged Composer console observations; no file I/O."""
import json,re,time
PREFIX=b"C3_FULL_MLM_PROGRESS_V8 "
class Progress:
    def __init__(self,clock=time.monotonic,emit=None):
        self.clock=clock;self.started=clock();self.buffer=b"";self.first=0;self.epochs=set();self.notices=0;self.metric_notices=0
        self.emit=emit or (lambda line:print(line.decode(),flush=True))
    def feed(self,data):
        lines=(self.buffer+data).split(bytes([10]))
        self.buffer=lines.pop()[-16384:]
        for line in lines:
            match=re.match(rb"^\[epoch=(\d+)\]\[batch=(\d+)/(\d+)\]",line.strip())
            if match:
                epoch,batch,total=map(int,match.groups())
                if self.first<8 or epoch not in self.epochs:
                    self.first+=1;self.epochs.add(epoch)
                    self._send(dict(kind="actual_author_console_progress",epoch_one_based=epoch,
                        batch_in_epoch=batch,dataloader_batches=total))
            elif self.first<=8 and self.metric_notices<24 and b"Train throughput/" in line:
                self.metric_notices+=1
                self._send(dict(kind="actual_author_console_metric",raw_author_metric=line.decode(errors="replace")[:512]))
    def _send(self,value):
        if self.notices>=48:return
        value.update(seconds_since_native_spawn_observation=self.clock()-self.started,
            synthetic_qualification=False,source="unchanged_Composer_console",full_budget_ETA_claimed=False)
        raw=PREFIX+json.dumps(value,sort_keys=True).encode()
        if len(raw)<=2048:self.emit(raw);self.notices+=1
class RootRelay:
    def __init__(self,emit=None):
        self.buffer=b"";self.sent=0;self.emit=emit or (lambda line:print(line.decode(),flush=True))
    def feed(self,data):
        lines=(self.buffer+data).split(bytes([10]))
        self.buffer=lines.pop()[-16384:]
        for line in lines:
            if line.startswith(PREFIX) and len(line)<=2048 and self.sent+len(line)<=65536:
                self.emit(line);self.sent+=len(line)
