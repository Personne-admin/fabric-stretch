#ifndef CONSTANTS_HH
#define CONSTANTS_HH

#include <stdint.h>

// Compile-time configuration of the stretch controller; see firmware/README.md for the full
// hardware description and serial protocol.
namespace Constants {

// Host side: fabsim.rig.hardware.protocol.BAUD (change both together).
constexpr int kSerialSpeed = 9600;
constexpr int kSerialTimeout = 25;

namespace Pin {
// Panic buttons (INPUT_PULLUP, falling edge): stop a running move, else jog 400 steps.
constexpr uint8_t IrqFirst = 2;
constexpr uint8_t IrqSecond = 3;

constexpr uint8_t RedLED = 4;
constexpr uint8_t GreenLED = 5;
constexpr uint8_t BlueLED = 6;

constexpr uint8_t MotorEnable = 7;
constexpr uint8_t MotorDir = 8;
constexpr uint8_t MotorStep = 9;

// Chip Select for the digi-pot.
constexpr uint8_t DigPotCS = 10;
// Master Out Slave In.
constexpr uint8_t Mosi = 11;

// LED switches (1-Wire bus), camera trigger output.
constexpr uint8_t OneWire = 14;
constexpr uint8_t Camera = 15;

// Load-cell clock and data lines, sampled by the timer ISR.
constexpr uint8_t SpiData = 16;
constexpr uint8_t SpiClk = 17;
} // namespace Pin

// Timer2 runs in CTC mode with prescaler 8 (0.5 us per count), so OCR2A = 63 gives a 32 us tick.
// The ISR toggles STEP on every tick; `Fraction` toggles (= 16 STEP pulses) make one command step,
// which equals one full motor step only if the driver is set to 1/16 microstepping
// (src/fabsim/rig/mechanics.py assumes this).
namespace Timer {
// Base OCR2A -> ~32 us.
constexpr uint8_t BaseOcr2a = 63;
// Divisor for motor steps.
constexpr uint8_t Fraction = 32;
} // namespace Timer

}; // namespace Constants

#endif /* CONSTANTS_HH */
