# Serial protocol: host and stretch controller

`motorV4` uses a newline-delimited USB serial protocol. Implementation:

- Firmware: `Commander.hh`, `Stepper.hh`, `Tensiometer.hh` in `firmware/motorV4/`.
- Host: `rig/hardware/protocol.py` (format/parser), `arduino.py` (I/O), `rig/dispatcher.py` (run logic).
- Simulator: `rig/hardware/fake.py`; tests: `tests/python/test_rig_protocol.py`.

Python paths are under `src/fabsim/`. See [Firmware](../firmware/README.md) for build and issues.

## Line settings

* 9600 baud, 8N1, no flow control (`protocol.BAUD`, `Constants::kSerialSpeed`).
* Opening the port resets the board (DTR). The host waits 2 s after opening
  (`ArduinoController.connect(settle_s=2.0)`), then discards pending input. The firmware prints
  its boot banner (below) during that time.
* Commands: one letter, an optional number, terminated by `\n`. The host sends `command + "\n"`
  and writes from several threads under a lock, so lines do not interleave. The firmware also
  ignores space, `\r` and tab between commands.
* The firmware reads numbers with `Serial.parseInt` and `Serial.parseFloat` and a 25 ms timeout.
  Send a command in a single write. `parseFloat` does not read exponent notation (`1e-3`), so the
  host always writes fixed point (`F12.500000,200,0`, `*0.01250000`).
* Replies: one text line each, terminated by `\n` (the host strips whitespace and ignores
  empty lines). Some lines are not answers to the last command (notifications).

## Commands

| Command | Meaning | Reply from the firmware |
| --- | --- | --- |
| `+N` | move `N` command steps in the loading direction. A new move replaces the one in progress | `jog: +N`, later `MOVE_DONE` |
| `-N` | move `N` steps in the release direction | `jog: -N`, later `MOVE_DONE` |
| `S` | stop the motor and disarm the force limit | `STOPPED remaining=<n>` |
| `L<mask>` | LED outputs, `mask` 0 to 255, bit `b` is output `b` | `leds: <mask> OK` (non-zero mask), `leds: 0` (zero mask) |
| `P<p>` | LED power 0 to 255 (potentiometer value `255 - p`) | `power: <p>` |
| `V<m>` | speed multiplier (timer period `max(64 / m, 2)` counts, integer division) | none |
| `#` | 20 ms pulse on the camera trigger pin | `trigger` |
| `@` | tare: store the last decoded load-cell value as zero (EEPROM) | `zero: <counts>` |
| `*<c>` | store the load-cell coefficient `c` in N per count (EEPROM); `c = 0` or unparsable is ignored | `coef: <c>` (none if ignored) |
| `t` | request force; wait asynchronously up to 30 s for the motor and 300 ms for a fresh frame | `TENSION force_n=<f> raw_counts=<i> zero_counts=<i> calibration_n_per_count=<f>` or `tension: N/A` |
| `F<limit>[,<steps>[,<dir>]]` | arm the overload trip (see [Force limit](#force-limit)); `F0` disarms | `LIMIT_ARMED limit_n=<f> release_motor_steps=<n> release_dir_positive=<0\|1>` or `limit: off` |
| `i` | liveness query | `status: API ready` |
| other | | `?<c>` |

`+N` and `-N` with `N = 0` do nothing and give no `MOVE_DONE`. Positive means loading.

## Notifications and boot output

| Line | Meaning |
| --- | --- |
| `MOVE_DONE` | a move ended normally. Not sent after a zero-length move, after a stop by a panic button, or after a force-limit trip. The auto-release move after a trip ends with one if its step count is positive |
| `MOVE_ABORTED reason=<reason> remaining=<n>` | interrupted move; reason is `host_stop`, `panic`, `force_limit` or `replaced` |
| `LIMIT_TRIP force_n=<f> raw_counts=<i> zero_counts=<i> calibration_n_per_count=<f> limit_n=<f>` | the armed limit tripped; the motor was stopped and the auto-release (if any) started |
| `OneWire search:` then one `3A:xx:xx:xx:00:00:00:xx` per line | boot: 1-Wire devices found (the LED switches) |
| `stretch-motor ready` | boot finished. The firmware prints no version |

The panic buttons (pins 2 and 3) can start or stop moves without any command. The host sees
`MOVE_DONE` only for moves that ended normally.

## Reply kinds in the host

`protocol.parse_reply(line)` returns a `Reply(kind, line, data)`. The kind is decided by prefix:

| Kind | Line |
| --- | --- |
| `move_done` | `MOVE_DONE` |
| `move_aborted` | `MOVE_ABORTED ...`; reason and remaining steps in `data` |
| `stopped` | `STOPPED ...`; remaining steps in `data` |
| `limit_trip` | starts with `LIMIT_TRIP` (also the old form `LIMIT_TRIP <force>`); `data` has the force fields |
| `limit_armed`, `limit_off` | `LIMIT_ARMED ...`, `limit: off` |
| `leds_ok` | `leds:` line containing `OK`; any other `leds:` line (`leds: 0`) is `leds_off` |
| `power`, `jog`, `coef`, `status`, `trigger` | `power:`, `jog:`, `coef:`, `status:`, `trigger` |
| `zero` | `zero: <int>`; `data["zero_counts"]` |
| `tension` | `TENSION ...` or the legacy `tension: <N>`; `data` has `force_n`, `raw_counts`, `zero_counts`, `calibration_n_per_count` (the last three `None` in the legacy form) |
| `banner` | ends with `stretch-motor ready` |
| `unknown_command` | `?` and one character |
| `other` | anything else (for example the 1-Wire addresses) |

`tension: N/A` is classified as `tension` with empty `data`, and the run records the force
reading as invalid.

## Move timing

Positive command steps load the cloth; negative steps release it. A command step takes 32 timer
ticks: 1.024 ms at `V1`, 0.512 ms at `V2`. A non-zero move runs `N + 1` periods. The host uses
that duration to set its completion deadline; physical distance is described under
[Motor and clamp](hardware.md#motor-and-clamp).

## Stopping

`S` disables the motor, clears pending completion and disarms the force limit, including during
auto-release. It cancels a pending tension request with `tension: N/A`, reports any interrupted
moves, then replies `STOPPED remaining=<n>` (also when already idle).
Panic buttons disarm the force limit too. Interrupted moves do not send `MOVE_DONE`.

The host sends `S` on cancellation or run failure and reports a missing acknowledgement.
Older firmware replies `?S` and cannot stop remotely; reflash before relying on this command.
`t` requests no longer block command parsing, though replies still share the serial line.

## Force limit

`F<limit>[,<release_steps>[,<dir>]]` arms a trip after three consecutive load-cell frames with
`abs(force) >= limit`. The trip stops the current move and starts an auto-release of
`release_steps` in direction `dir` (`1` positive, `0` negative). `F0` disarms it.
The trip sends `LIMIT_TRIP`; only a non-zero auto-release move subsequently sends `MOVE_DONE`.

## Firmware caveats

See [firmware issues](../firmware/README.md#known-issues) before relying on acknowledgements
or command responsiveness.

Change `Constants::kSerialSpeed` and host `protocol.BAUD` together if changing the baud rate.
