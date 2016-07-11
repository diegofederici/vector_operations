#include <time.h>
#include <cmath>
#include <gtest/gtest.h>
#include <chrono>

#include "vector_operations.h"

struct ScopedTimer {
  ScopedTimer(const string& name_) : name(name_), start(chrono::steady_clock::now()) {}

  void Output() {
    chrono::steady_clock::time_point clock_end = chrono::steady_clock::now();
    chrono::steady_clock::duration time_span = clock_end - start;
    double seconds = double(time_span.count()) * chrono::steady_clock::period::num 
      / (chrono::steady_clock::period::den);
    cout << name << ": " << setprecision(9) << 1000.0 * seconds << "ms\n";
  }
    

  string name;
  chrono::steady_clock::time_point start;
};

vector<float> CreateData(const int size) {
  vector<float> v(size);
  float i = 1;
  for (int j = 0; j < 2 * (size / 2); j += 2) {
    v[j] = i;
    v[j+1] = i;
    i *= -1;
  }
  return v;
}

void TestDotProd(float(*f)(const float *, const float *, const size_t),
		 const vector<float>& data, const string& name) {
  ScopedTimer t("  - TestDotProd " + name);
  for (int i = data.size() / 2; i < data.size() - 1; ++i) {
    const float dot = f(data.data(), data.data() - 1, i);
    const float exp_dot = (data[i] * data[i - 1] == -1) ? 1 : 0;
    if (fabs(dot - exp_dot) > 0.001) {
      cerr << "error at " << i << ": expected " << exp_dot << " got " << dot << endl;
      return;
    }
  }
  t.Output();
}

void TestDotProdF16(float(*f)(const uint16_t *, const uint16_t *, const size_t),
		    const vector<uint16_t>& data, const string& name) {
  ScopedTimer t("  - TestDotProd " + name);
  for (int i = data.size() / 2; i < data.size() - 1; ++i) {
    const float dot = f(data.data(), data.data() - 1, i);
    const float exp_dot = 
      (half8::MultiplyOne(data[i], data[i - 1]) == -1) ? 1 : 0;
    if (fabs(dot - exp_dot) > 0.001) {
      cerr << "error at " << i << ": expected " << exp_dot << " got " << dot << endl;
      return;
    }
  }
  t.Output();
}


int main(int argc, char **argv) {
  for (int size = 25000; size < 200 * 1000; size += 25000) {
    vector<float> data = CreateData(size);
    cout << "- Testing vector with maximum size of " << size << endl;
    TestDotProd(naive::DotProduct, data,         "naive      ");
    TestDotProd(DotProduct<float4>, data,        "SSE        ");
    TestDotProd(DotProduct<float8>, data,        "AVX        ");

    auto half_data = half8::FloatToFp16(data);
    TestDotProdF16(DotProduct<half8>, half_data, "AVX F16C   ");
  }
}
