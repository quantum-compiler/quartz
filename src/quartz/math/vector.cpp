#include "vector.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

namespace quartz {
bool Vector::apply_matrix(MatrixBase *mat,
                          const std::vector<int> &qubit_indices) {
  const int n0 = (int)qubit_indices.size();
  assert(n0 <= 30);  // 1 << n0 does not overflow
  assert(mat->size() == (1 << n0));
  const int S = (int)data_.size();
  assert(S >= (1 << n0));

  std::vector<ComplexType> buffer(1 << n0);

  for (int i = 0; i < S; i++) {
    bool already_applied = false;
    for (const auto &j : qubit_indices) {
      if (i & (1 << j)) {
        already_applied = true;
        break;
      }
    }
    if (already_applied)
      continue;

    for (auto &val : buffer) {
      val = ComplexType(0);
    }

    // matrix * vector
    for (int j = 0; j < (1 << n0); j++) {
      for (int k = 0; k < (1 << n0); k++) {
        int index = i;
        for (int l = 0; l < n0; l++) {
          if (k & (1 << l)) {
            index ^= (1 << qubit_indices[l]);
          }
        }
        buffer[j] += (*mat)[j][k] * data_[index];
      }
    }

    for (int k = 0; k < (1 << n0); k++) {
      int index = i;
      for (int l = 0; l < n0; l++) {
        if (k & (1 << l)) {
          index ^= (1 << qubit_indices[l]);
        }
      }
      data_[index] = buffer[k];
    }
  }
  return true;
}

void Vector::print() const {
  std::cout << "[" << std::setprecision(4);
  for (int i = 0; i < size(); i++) {
#ifdef USE_ARBLIB
    data_[i].print(4);
#else
    std::cout << data_[i];
#endif
    if (i != size() - 1)
      std::cout << " ";
  }
  std::cout << "]" << std::endl;
}

Vector Vector::random_generate(int num_qubits, std::mt19937 *gen) {
  // Standard mersenne_twister_engine seeded with 0
  static std::mt19937 static_gen(0);
  if (!gen) {
    gen = &static_gen;
  }
  static std::uniform_int_distribution<int> dis_int(0, 1);
  assert(num_qubits <= 30);
  Vector result(1 << num_qubits);
  unsigned int remaining_numbers = (2u << num_qubits);

#ifdef USE_ARBLIB
  constexpr slong kRandPrec = 64;
  class FlintRandWrapper {
   public:
    FlintRandWrapper() { flint_randinit(state); }
    ~FlintRandWrapper() { flint_randclear(state); }
    flint_rand_t state{};
  };
  static FlintRandWrapper flint_rand;
  arb_t remaining_norm;
  arb_init(remaining_norm);
  arb_one(remaining_norm);  // remaining_norm = 1;
  arb_t number;
  arb_init(number);

  auto generate = [&]() {
    arb_t tmp;
    arb_init(tmp);
    if (remaining_numbers > 1) {
      // Same algorithm as WeChat red packet
      /*  // This method seems not generating uniform distribution
      fmpq_t random_value;
      arb_unit_interval(tmp);
      arb_get_rand_fmpq(random_value, flint_rand_state, tmp,
      kRandPrec); arb_set_fmpq(number, random_value, kRandPrec);
       */
      arb_urandom(number, flint_rand.state, kRandPrec);
      // number = random value in [0, 1] now
      arb_mul(tmp, number, remaining_norm, kRandPrec);
      arb_set(number, tmp);  // number *= remaining_norm;
      arb_div_si(tmp, number, remaining_numbers * 2, kRandPrec);
      arb_set(number, tmp);  // number /= remaining_numbers * 2;
    } else {
      arb_set(number, remaining_norm);  // number = remaining_norm;
    }
    remaining_numbers--;
    arb_sub(tmp, remaining_norm, number, kRandPrec);
    arb_set(remaining_norm, tmp);  // remaining_norm -= number;
    arb_sqrt(tmp, number, kRandPrec);
    arb_set(number, tmp);  // number = std::sqrt(number);
    if (dis_int(*gen)) {
      arb_neg(tmp, number);
      arb_set(number, tmp);  // number = -number;
    }
    arb_clear(tmp);
    return number;
  };
#else
  static std::uniform_real_distribution<ComplexType::value_type> dis_real(0, 1);
  ComplexType::value_type remaining_norm = 1;

  auto generate = [&]() {
    auto number = remaining_norm;
    if (remaining_numbers > 1) {
      // Same algorithm as WeChat red packet
      number = dis_real(*gen) * (remaining_norm / remaining_numbers * 2);
    }
    remaining_numbers--;
    remaining_norm -= number;
    number = std::sqrt(number);
    if (dis_int(*gen)) {
      number = -number;
    }
    return number;
  };
#endif

  for (int i = 0; i < (1 << num_qubits); i++) {
    result[i].real(generate());
    result[i].imag(generate());
  }

#ifdef USE_ARBLIB
  arb_clear(remaining_norm);
  arb_clear(number);
#endif

  return result;
}

ComplexType Vector::dot(const Vector &other) const {
  assert(size() == other.size());
  ComplexType result(0);
  for (int i = 0; i < size(); i++) {
    result += data_[i] * other[i];
  }
  return result;
}

}  // namespace quartz
