// nvml.h — мощность карт через NVML (libnvidia-ml.so.1 грузится dlopen: сборка не зависит от NVML,
// без драйвера/библиотеки — просто нет данных). Сначала nvmlDeviceGetPowerUsage; если карта его не
// отдаёт (RTX 4060 на драйвере 595: «N/A»), — среднее по выборкам NVML_TOTAL_POWER_SAMPLES (то же,
// что «Power Samples» в nvidia-smi -q -d POWER).
#pragma once
#include <dlfcn.h>
#include <cstdint>
#include <cstring>

namespace nvml {

typedef void *Dev;
union Value { double d; unsigned ui; unsigned long ul; unsigned long long ull; long long sll; };
struct Sample { unsigned long long ts; Value v; };
enum { TOTAL_POWER_SAMPLES = 0 };
enum { VT_DOUBLE = 0, VT_UINT = 1, VT_ULONG = 2, VT_ULLONG = 3, VT_SLLONG = 4 };

struct Lib {
  void *h = nullptr;
  int (*init)() = nullptr;
  int (*by_bus)(const char *, Dev *) = nullptr;
  int (*power)(Dev, unsigned *) = nullptr;
  int (*samples)(Dev, int, unsigned long long, int *, unsigned *, Sample *) = nullptr;
  bool ok = false;
  Lib() {
    h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    init = (int (*)())dlsym(h, "nvmlInit_v2");
    by_bus = (int (*)(const char *, Dev *))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    power = (int (*)(Dev, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
    samples = (int (*)(Dev, int, unsigned long long, int *, unsigned *, Sample *))dlsym(h, "nvmlDeviceGetSamples");
    ok = init && by_bus && power && init() == 0;
  }
};
inline Lib &lib() { static Lib l; return l; }

// мощность одной карты; bus — строка cudaDeviceGetPCIBusId ("0000:01:00.0")
struct Power {
  Dev dev = nullptr;
  unsigned long long last = 0;  // метка последней выборки
  bool init(const char *bus) {
    Lib &l = lib();
    return l.ok && l.by_bus(bus, &dev) == 0;
  }
  // ватты или < 0, если данных нет
  double watts() {
    Lib &l = lib();
    if (!dev) return -1;
    unsigned mw = 0;
    if (l.power(dev, &mw) == 0 && mw > 0) return mw / 1000.0;
    if (!l.samples) return -1;
    static thread_local Sample buf[256];
    unsigned cnt = 256;
    int vt = 0;
    if (l.samples(dev, TOTAL_POWER_SAMPLES, last, &vt, &cnt, buf) != 0 || cnt == 0) return -1;
    double sum = 0;
    for (unsigned i = 0; i < cnt; i++) {
      const Value &v = buf[i].v;
      sum += vt == VT_DOUBLE ? v.d : vt == VT_UINT ? v.ui : vt == VT_ULONG ? (double)v.ul
           : vt == VT_ULLONG ? (double)v.ull : (double)v.sll;
      if (buf[i].ts > last) last = buf[i].ts;
    }
    return sum / cnt / 1000.0;  // мВт → Вт
  }
};

}  // namespace nvml
