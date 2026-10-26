# Independent process preparation and rollout review

The process encoder preserves order by yielding submitted chunk futures in native tree order. It never returns completion order. Each chunk contains whole trees and their exact observed sites. The byte splitter sends an oversized individual tree to the parent's existing serial route after earlier futures have drained. It does not prune that tree or alter the marker chunk reduction inside the branch support calculation.

The queue contains no more than the requested worker count of futures. Its admission separately charges per-worker runtime and bounded input/output transport. The currently delivered chunk may coexist with the pending queue, and the transport allowance includes the extra slot. Pickle transport receives NumPy slices, which serialize the slice contents rather than the complete parent matrix. Worker code uses the original support kernel on each complete tree. Ordered parent assembly, donor frequencies and bank construction remain unchanged.

Worker initialization applies one Torch and native-library thread inside each spawned child and verifies that CUDA has not initialized. It does not change the parent thread settings. An independent fresh process captured Python, NumPy and Torch RNG plus thread settings before importing the process candidate and before installing the existing hooks. All captured state remained unchanged, with no CUDA initialization. Evidence is `process_import_audit.json`.

The initial full-process attempt exposed a worker guard issue: Linux historical `ru_maxrss` can retain the launching process's pre-exec high-water value. The implementation agent corrected the worker guard to use current RSS and retained historical high-water values only as telemetry. Final parity and full-parent evidence must be regenerated for that source. Runtime RSS checks are sampled guards, not an operating-system hard memory limit, and must not be described as such.

Failure handling cancels outstanding queued futures and the process-pool context joins owned workers. Already-running bounded chunks finish before context exit. This avoids detached processes, but does not promise instantaneous cancellation of an executing chunk. The external campaign timeout remains necessary for a process that stops responding. Existing parity tests cover worker exceptions and denied admission. The complete observed-parent signature must match the reference for both all tensor bytes and biological metadata before release.

## Current queue boundary

The active repair API only permits FAILED or BLOCKED tasks. It cannot replace a healthy running prediction or an ordinary pending task. The continuation API rejects live or unknown task rows and requires the previous controller and session to be absent. A plan-file edit or inventing a failure would bypass those safeguards.

The healthy European prediction should therefore finish with its existing pinned implementation. The new explicit prediction wrapper is available for a legitimate future repair or reviewed continuation. A pending-task upgrade while a controller is alive requires a separate explicit scheduler mechanism and safe migration. It is outside the existing API and is not accomplished by writing a replacement plan file.
