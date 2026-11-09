#include <cassert>
#include <limits>
#include <string>

#include "Arduino.h"

// Keep the real command parser and stepper; substitute sensors and outputs.
#define TENSIOMETER_HH
#define CAMERA_HH
#define LIGHT_HH
namespace Camera { void trigger(int) {} }
namespace Lights { void set_mask(uint8_t) {} void set_power(uint8_t) {} }
namespace Tension {
namespace Detail {
volatile bool limit_active = false, tripped = false, unread_isr_data = false;
volatile uint8_t last_frame_bits = 24;
volatile float limit_force_n = 0, tripped_force = 0;
volatile int16_t tripped_raw_counts = 0;
int16_t zero = 0;
float coef = 0.0125f;
int32_t auto_release_steps = 0;
bool release_dir_positive = false;
}
int16_t raw = 0;
bool ready = false;
int16_t raw_counts() { return raw; }
float force_newtons_from_raw(int16_t value) { return Detail::coef * (Detail::zero - value); }
float tripped_force() { return Detail::tripped_force; }
bool is_ready() { return ready; }
void clear_ready() { ready = false; }
void tare() { Detail::zero = raw; }
void set_multiplier(float value) { Detail::coef = value; }
void disarm_force_limit() { Detail::limit_active = false; Detail::tripped = false; }
void set_force_limit(float limit, int32_t release, bool direction) {
  Detail::limit_active = true;
  Detail::limit_force_n = limit;
  Detail::auto_release_steps = release;
  Detail::release_dir_positive = direction;
}
}

#include "../motorV4/Commander.hh"

void command(const std::string& text) {
  Serial.input = text + "\n";
  Serial.position = 0;
  while (Serial.available()) Command::poll();
}
std::string output() { auto text = Serial.output.str(); Serial.output.str(""); return text; }
void ticks(unsigned count) { while (count--) Stepper::isr_callback(); }

int main() {
  command("+100");
  command("F1,200,0");
  output();
  command("t");
  assert(Command::tension_pending && Stepper::is_busy());
  command("S");
  auto text = output();
  assert(text.find("MOVE_ABORTED reason=host_stop remaining=100") != std::string::npos);
  assert(text.find("STOPPED remaining=100") != std::string::npos);
  assert(text.find("tension: N/A") != std::string::npos);
  assert(!Command::tension_pending && !Stepper::is_busy());
  assert(!Tension::Detail::limit_active);
  assert(pins[Constants::Pin::MotorEnable] == HIGH);
  ticks(1000);
  Stepper::poll();
  assert(output().empty());

  command("S");
  assert(output() == "STOPPED remaining=0\n");
  command("+2");
  ticks(32 * 3);
  Stepper::poll();
  text = output();
  assert(text.find("MOVE_DONE") != std::string::npos);
  assert(text.find("MOVE_ABORTED") == std::string::npos);

  command("+50");
  output();
  command("F1,200,0");
  SREG = 0;
  Stepper::stop_from_isr();
  assert(SREG == 0 && !Tension::Detail::limit_active);
  interrupts();
  Stepper::poll();
  text = output();
  assert(text.find("reason=panic") != std::string::npos);
  assert(text.find("MOVE_DONE") == std::string::npos);

  command("+50");
  command("-10");
  Stepper::poll();
  assert(output().find("reason=replaced") != std::string::npos);
  assert(Stepper::is_busy());
  command("S");
  output();

  command("F1,200,0");
  command("+100");
  Tension::raw = -1000;
  for (int i = 0; i < 3; ++i) {
    Tension::Detail::unread_isr_data = true;
    Stepper::check_force_limit_from_isr();
  }
  assert(Stepper::is_busy()); // Auto-release can be stopped before the trip is reported.
  command("S");
  text = output();
  assert(text.find("reason=force_limit") != std::string::npos);
  assert(text.find("reason=host_stop") != std::string::npos);
  assert(text.find("LIMIT_TRIP") != std::string::npos);
  assert(!Stepper::is_busy());

  // Stop must suppress a completion that the main loop has not yet emitted.
  command("+1");
  ticks(64);
  command("S");
  assert(output().find("MOVE_DONE") == std::string::npos);

  command("t");
  assert(Command::tension_pending);
  now_ms = 301;
  Command::poll();
  assert(output() == "tension: N/A\n");
  command("t");
  Tension::ready = true;
  Command::poll();
  assert(output().find("TENSION force_n=") != std::string::npos);

  now_ms = std::numeric_limits<uint32_t>::max() - 100;
  command("+100");
  command("t");
  now_ms = 30000;
  Command::poll();
  assert(!Command::tension_pending); // Timeout survives millis() wraparound.
}
