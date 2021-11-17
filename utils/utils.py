import math
import z3

meta_index_num_qubits = 0
meta_index_num_input_parameters = 1
meta_index_num_total_parameters = 2
meta_index_num_gates = 3
meta_index_other_hash_values = 4
meta_index_original_fingerprint = 5

kCheckPhaseShiftOfPiOver4Index = 10000  # A copy of the value in utils.h

kPhaseFactorCoeffs = [0, 1, -1, 2, -2]
kPhaseFactorConstant = math.pi / 4.0
kPhaseFactorConstantCoeffMin = 0
kPhaseFactorConstantCoeffMax = 7
kPhaseFactorConstantCosTable = [1, 1.0 / z3.Sqrt(2), 0, -1.0 / z3.Sqrt(2), -1, -1.0 / z3.Sqrt(2), 0, 1.0 / z3.Sqrt(2)]
kPhaseFactorConstantSinTable = [0, 1.0 / z3.Sqrt(2), 1, 1.0 / z3.Sqrt(2), 0, -1.0 / z3.Sqrt(2), -1, -1.0 / z3.Sqrt(2)]
kPhaseFactorEpsilon = 1e-6
