# Stretch controller firmware (`motorV4`)

Arduino controller for the motor, LEDs and load cell. The host is `fabsim rig`;
commands and replies are in the [protocol reference](../docs/serial_protocol.md).

## LED switches

The four 1-Wire switch addresses are stored in `motorV4/Light.hh`. Update them if a switch is
replaced; the boot output lists devices found on the bus. Each chip serves two outputs.
Assign those outputs to lamps in the app's *Identify LEDs* dialog; see
[Lights](../docs/hardware.md#lights).

## Libraries and board

Required libraries:

* `OneWire`
* `U8g2` (provides `U8x8lib.h`)
* `digitalWriteFast`
* built in: `SPI`, `Wire`, `EEPROM`

FQBN: `arduino:avr:uno`. A Nano works with `arduino:avr:nano:cpu=atmega328` (or
`...:atmega328old` for the old bootloader).

## Build and flash

With the Arduino IDE: open `motorV4/motorV4.ino`, select the board and port, upload. With
`arduino-cli`:

```sh
arduino-cli core install arduino:avr
arduino-cli lib install OneWire U8g2 digitalWriteFast
arduino-cli compile --fqbn arduino:avr:uno firmware/motorV4
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyUSB0 firmware/motorV4   # COM3 on Windows
```

After flashing a fresh board, [calibrate the load cell](../docs/hardware.md#load-cell).
Calibration is stored in EEPROM; see [Blank EEPROM](#blank-eeprom) before the first run.

## Controller interface

Pin assignments: [`motorV4/Constants.hh`](motorV4/Constants.hh).
Rig setup and calibration: [Hardware](../docs/hardware.md).

## EEPROM layout

| Address | Type | Content |
| --- | --- | --- |
| 80 | `int16` | tare: zero counts (`@`) |
| 84 | `float` (4 bytes) | load-cell coefficient in N per count (`*`) |

## Known issues

### No stop command

Stop ends acquisition on the PC; a commanded move continues until completion, a panic button
or a force-limit trip. The host times out waiting for `MOVE_DONE`. Add an `S` command replying
`STOPPED remaining=<n>`, and `MOVE_ABORTED` for interrupted moves.

### Unchecked LED acknowledgement

`Lights::set_mask` prints `OK` even if the 1-Wire write failed. Check `write_gpio` results and
reply `FAIL dev=<i>` on failure. The host currently has to judge image brightness.

### Interrupt safety

The panic-button ISRs call `Stepper::move`, which calls `interrupts()` inside the handler.
They should use `move_from_isr`. `check_force_limit_from_isr` also does floating-point work
inside the Timer2 ISR. It may overrun the tick, introduce step jitter and lose bit-banged
load-cell frames; the ISR duration has not been measured.

`Tension::Detail::value` is written by `raw_counts()` from both the ISR and `loop()` without
masking. Use a local variable there. A further fix would copy frames in the ISR and convert
in `loop()`, or compare integer counts against a precomputed threshold.

### Display redraws

`Display::update()` redraws over I2C on every `loop()` iteration while a frame is ready. Only
`t` clears that flag. Rate-limiting the display to about 10 Hz would reduce command-loop delays.

### Force-limit arm refusal

`F` replies `LIMIT_ARMED` even when the coefficient is zero and arming is refused.
Reply `limit: refused` instead; the host currently cannot tell.

### Blank EEPROM

`Tension::load_config` copies the tare and coefficient from EEPROM without checking them.
A fresh chip contains `0xFF`, which produces a NaN coefficient. Set the coefficient before
using a new board; see [Load cell calibration](../docs/hardware.md#load-cell).

### Blocking tension reads

`t` busy-waits for the motor for up to 30 s, during which no further command is parsed.
A `TENSION` line also takes about 100 ms to transmit at the current baud rate.
A non-blocking implementation and a change to 115200 baud would help. A baud-rate change must update
both sides as described in [Line settings](../docs/serial_protocol.md#line-settings).

### Extra motor period

A non-zero move of `N` command steps runs `N + 1` periods. The host accounts for the extra
step in `displacement_mm` (`mechanics.travelled_steps`) and in the move deadline.
`session.json` records both `requested_displacement_um_per_increment` and
`travelled_displacement_um_per_increment`. The distance of a command step depends on the
[configured motor mechanics](../docs/hardware.md#motor-and-clamp).

### Hard-coded values

The source fixes the four 1-Wire addresses (`Light.hh`), the initial coefficient (0.0125 N/count,
overwritten from EEPROM), panic-button jogs, camera pulse length, tension-read timeouts,
trip confirmation count and minimum load-cell frame length. Changes currently require a rebuild.
The command timings and trip conditions are in the [protocol reference](../docs/serial_protocol.md).

### Unused code

`CurrentSensor::read_current` is never called although the INA219 is initialised. Other unused
members are `Camera::is_busy`, `Tension::get_ready_count`, `was_tripped`, `clear_tripped`,
`Lights::Detail::kFamilyId` and `Pin::Mosi`.
The OLED splash says "stretch-motor 2" while the serial banner says `stretch-motor`.

### Missing firmware version

Add `fw: motorV4 <git>` to the boot banner and `i` reply so sessions can record the version.
