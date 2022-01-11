#include "math/vector.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <random>

using namespace quartz;

bool Vector::apply_matrix(MatrixBase *mat,
                          const std::vector< int > &qubit_indices) {
	const int n0 = (int)qubit_indices.size();
	assert(n0 <= 30); // 1 << n0 does not overflow
	assert(mat->size() == (1 << n0));
	const int S = (int)data_.size();
	assert(S >= (1 << n0));

	std::vector< ComplexType > buffer(1 << n0);

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

Vector Vector::random_generate(int num_qubits) {
	// Standard mersenne_twister_engine seeded with 0
#ifdef USE_ARBLIB
	using generator_value_type = double;
#else
	using generator_value_type = ComplexType::value_type;
#endif
	static std::mt19937 gen(0);
	static std::uniform_real_distribution< generator_value_type > dis_real(0,
	                                                                       1);
	static std::uniform_int_distribution< int > dis_int(0, 1);

	Vector result(1 << num_qubits);
	generator_value_type remaining_norm = 1;
	int remaining_numbers = (2 << num_qubits);

	auto generate = [&]() {
		auto number = remaining_norm;
		if (remaining_numbers > 1) {
			// Same algorithm as WeChat red packet
			number = dis_real(gen) * (remaining_norm / remaining_numbers * 2);
		}
		remaining_numbers--;
		remaining_norm -= number;
		number = std::sqrt(number);
		if (dis_int(gen)) {
			number = -number;
		}
		return number;
	};

	for (int i = 0; i < (1 << num_qubits); i++) {
		result[i].real(generate());
		result[i].imag(generate());
	}
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
