#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#define F(value) value
constexpr int HIGH = 1;
constexpr int LOW = 0;
inline uint8_t SREG = 0x80;
inline uint8_t TCCR2A, TCCR2B, TCNT2, TIMSK2, OCR2A;
constexpr int WGM21 = 1, CS21 = 1, OCIE2A = 1;
inline int pins[32] = {};
inline uint32_t now_ms = 0;
inline uint32_t millis() { return now_ms; }
inline void noInterrupts() { SREG &= 0x7f; }
inline void interrupts() { SREG |= 0x80; }
inline void digitalWriteFast(int pin, int value) { pins[pin] = value; }
inline void digitalToggleFast(int pin) { pins[pin] ^= 1; }

struct TestSerial {
  std::string input;
  std::ostringstream output;
  size_t position = 0;
  int available() const { return static_cast<int>(input.size() - position); }
  int read() { return available() ? input[position++] : -1; }
  int peek() const { return available() ? input[position] : -1; }
  long parseInt() {
    char* end;
    const char* start = input.c_str() + position;
    long result = std::strtol(start, &end, 10);
    position += end - start;
    return result;
  }
  float parseFloat() {
    char* end;
    const char* start = input.c_str() + position;
    float result = std::strtof(start, &end);
    position += end - start;
    return result;
  }
  template<class T> void print(T value) { output << value; }
  void print(float value, int precision) { output << std::fixed << std::setprecision(precision) << value; }
  template<class T> void println(T value) { print(value); output << '\n'; }
  void println(float value, int precision) { print(value, precision); output << '\n'; }
};
inline TestSerial Serial;
