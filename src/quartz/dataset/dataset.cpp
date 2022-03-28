#include "dataset.h"
#include "equivalence_set.h"

#include <vector>
#include <unordered_map>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <mutex>
#include "omp.h"

namespace quartz
{
    int Dataset::num_hash_values() const { return (int) dataset.size(); }

    int Dataset::num_total_dags() const {
        int ret = 0;
        for (const auto &it: dataset) {
            ret += (int) it.second.size();
        }
        return ret;
    }

    bool Dataset::save_json(Context *ctx, const std::string &file_name) const {
        std::ofstream fout;
        fout.open(file_name, std::ofstream::out);
        if (!fout.is_open()) {
            return false;
        }

        fout << "[" << std::endl;

        // The generated parameters for random testing.
        auto all_parameters = ctx->get_all_generated_parameters();
        fout << "[";
        bool start0 = true;
        for (auto &param: all_parameters) {
            if (start0) {
                start0 = false;
            } else {
                fout << ", ";
            }
            fout << std::scientific << std::setprecision(17) << param;
        }
        fout << "]," << std::endl;

        fout << "{" << std::endl;
        start0 = true;
        for (const auto &it: dataset) {
            if (it.second.empty()) {
                // Empty DAG set
                continue;
            }
            if (start0) {
                start0 = false;
            } else {
                fout << ",";
            }
            fout << "\"" << std::hex << it.first << "\": [" << std::endl;
            bool start = true;
            for (const auto &dag: it.second) {
                if (start) {
                    start = false;
                } else {
                    fout << ",";
                }
                fout << dag->to_json();
            }
            fout << "]" << std::endl;
        }
        fout << "}" << std::endl;

        fout << "]" << std::endl;
        return true;
    }

    int Dataset::remove_singletons(Context *ctx) {
        int num_removed = 0;
        for (auto it = dataset.begin(); it != dataset.end();) {
            if (it->second.size() != 1) {
                it++;
                continue;
            }
            auto &dag = it->second[0];
            auto it_hash_value = dag->hash(ctx);
            bool found_possible_equivalence = false;
            for (auto &hash_value: dag->other_hash_values()) {
                auto find_other = dataset.find(hash_value);
                if (find_other != dataset.end() &&
                    !find_other->second.empty()) {
                    found_possible_equivalence = true;
                    break;
                }
                assert(hash_value ==
                       it_hash_value + 1); // Only deal with this case...
            }
            // ...so that we know for sure that only DAGs with hash value equal
            // to |it_hash_value - 1| can have other_hash_values() containing
            // |it_hash_value|.
            auto find_other = dataset.find(it_hash_value - 1);
            if (find_other != dataset.end() && !find_other->second.empty()) {
                found_possible_equivalence = true;
                break;
            }
            if (found_possible_equivalence) {
                it++;
                continue;
            }
            // Remove |it|.
            auto remove_it = it;
            it++;
            dataset.erase(remove_it);
            num_removed++;
        }
        return num_removed;
    }

    int Dataset::normalize_to_minimal_circuit_representations(Context *ctx) {
        int num_removed = 0;
        std::vector<std::unique_ptr<DAG> > dags_to_insert_afterwards;
        auto dag_already_exists =
                [](const DAG &dag,
                   const std::vector<std::unique_ptr<DAG> > &new_dags) {
                    for (auto &other_dag: new_dags) {
                        if (dag.fully_equivalent(*other_dag)) {
                            return true;
                        }
                    }
                    return false;
                };

        for (auto &item: dataset) {
            auto &current_hash_value = item.first;
            auto &dags = item.second;
            auto size_before = dags.size();
            std::vector<std::unique_ptr<DAG> > new_dags;
            std::unique_ptr<DAG> new_dag;

            for (auto &dag: dags) {
                bool is_minimal = dag->minimal_circuit_representation(&new_dag);
                if (!is_minimal) {
                    if (!dag_already_exists(*new_dag, new_dags)) {
                        new_dags.push_back(std::move(new_dag));
                    }
                    dag = nullptr; // delete the original DAG
                }
            }
            if (!new_dags.empty()) {
                // |item| is modified.
                for (auto &dag: dags) {
                    // Put all dags into |new_dags|.
                    if (dag != nullptr) {
                        if (!dag_already_exists(*dag, new_dags)) {
                            new_dags.push_back(std::move(dag));
                        }
                    }
                }
                // Update |dags|.
                dags.clear();
                for (auto &dag: new_dags) {
                    const auto hash_value = dag->hash(ctx);
                    if (hash_value == current_hash_value) {
                        dags.push_back(std::move(dag));
                    } else {
                        // The hash value changed due to floating-point errors.
                        // Insert |dag| later to avoid corrupting the iterator
                        // of |dataset|.
                        dags_to_insert_afterwards.push_back(std::move(dag));
                    }
                }
                auto size_after = dags.size();
                num_removed += (int) (size_before - size_after);
            }
        }
        for (auto &dag: dags_to_insert_afterwards) {
            const auto hash_value = dag->hash(ctx);
            if (!dag_already_exists(*dag, dataset[hash_value])) {
                num_removed--; // Insert |dag| back.
                dataset[hash_value].push_back(std::move(dag));
            }
        }
        return num_removed;
    }

    bool Dataset::insert(Context *ctx, std::unique_ptr<DAG> dag) {
        const auto hash_value = dag->hash(ctx);
        bool ret;
        static std::mutex lock_dataset; // TODO Colin : replace it with OMP "lock"
        {
            std::lock_guard<std::mutex> lg_dataset(lock_dataset);
            ret = dataset.count(hash_value) == 0;
            dataset[hash_value].push_back(std::move(dag));
        }
        return ret;
    }

    void Dataset::clear() {
        // Caveat here: if only dataset.clear() is called, the behavior will be
        // different with a brand new Dataset.
        dataset = std::unordered_map<DAGHashType,
                std::vector<std::unique_ptr<
                        DAG> > >();
    }

    void Dataset::find_equivalences(Context *ctx) {
        /*
         * iterate over every (hashtag, dags), check whether the dags are really equivalent
         * get info like {'hashtag_index_1': [dags], ...}
         * each key is mapped to a list containing real equivalent dags
         * build ec_classes: [ class_0: [dag_0, dag_1], ... ]
         */
        // convert unordered_map obj dataset to vector to enable parallelization
        std::vector<std::pair< DAGHashType, std::vector<std::unique_ptr<DAG>>* >> vec_dataset;
        vec_dataset.reserve(dataset.size());
        for (auto& [hash, dags] : dataset) {
            vec_dataset.emplace_back(hash, &dags);
        }
        std::vector< std::vector<EquivClassTag> > class_hashes(dataset.size());
        std::vector< std::vector<std::unique_ptr<EquivalenceClass>> > classes(dataset.size());

        #pragma omp parallel for default(none) shared(vec_dataset, class_hashes, classes)
        for (size_t i = 0; i < vec_dataset.size(); i++) {
            // build vector<EquivalenceClass> at classes[i]
            DAGHashType hashtag = vec_dataset[i].first;
            std::vector<std::unique_ptr<DAG>>* dags = vec_dataset[i].second;
            // more than one dag, need to check if they are really equivalent
            std::vector<EquivClassTag>& this_class_hashes = class_hashes[i];
            std::vector<std::unique_ptr<EquivalenceClass>>& this_eccs = classes[i];
            auto insert_ecc_with_dag = [&](const std::unique_ptr<DAG>& dag) {
                this_class_hashes.emplace_back(hashtag, this_class_hashes.size());
                auto ecc = std::make_unique<EquivalenceClass>();
                // ATTENTION : copy construct a dag
                ecc->insert(std::make_unique<DAG>(*dag));
                this_eccs.emplace_back(std::move(ecc));
            };
            for (const auto& dag : *dags) {
                if (this_class_hashes.empty()) {
                    // the first one, no need to check equivalence with others
                    insert_ecc_with_dag(dag);
                }
                else {
                    // check equivalence between dag and inserted ones
                    bool found_equivalent_one = false;
                    for (auto& inserted_ecc : this_eccs) {
                        DAG* other_dag = inserted_ecc->get_representative();
                        if (true /* TODO Colin : equivalent(dag, other_dag, parameters_for_fingerprint) */) {
                            found_equivalent_one = true;
                            // ATTENTION : copy construct a dag
                            inserted_ecc->insert(std::make_unique<DAG>(*dag));
                            break;
                        }
                    } // end for this_eccs
                    if (!found_equivalent_one) {
                        // non-equivalent dag with the same hash, build a new ecc
                        insert_ecc_with_dag(dag);
                    }
                }
            } // end for dags
        } // end for vec_dataset (paralleled)


        /*
         * find equivalences with different hash
         * 1. iterate over all (hashtag, dags)in class_hashes, classes
         * 2. build a dict: (other_hashtags) { other_hash: { phase: [dags] } }
         * 3. build dags_to_verify and do the verification
         * 4. store the result in equiv_edges
         */
        // ATTENTION Colin : Is PairHash safe?
        std::unordered_map< EquivClassTag, std::vector<EquivClassTag>, PairHash > equiv_edges;
        for (size_t i = 0; i < classes.size(); i++) {
            for (size_t j = 0; j < classes[i].size(); j++) {
            // iterate over each ecc
                const EquivClassTag hashtag = class_hashes[i][j];
                const EquivalenceClass* const ecc = classes[i][j].get();
                const std::vector<DAG*> dags = ecc->get_all_dags();
                /*
                 * |other_hashtags[other_hash][None]| indicates if it's possible that a DAG with |other_hash|
                 *      is equivalent with a DAG with |hashtag| without phase shifts.
                 * |other_hashtags[other_hash][phase_shift_id]| is a list of DAGs with |hashtag| that can be equivalent
                 *      to a DAG with |other_hash| under phase shift |phase_shift_id|.
                 */
                using PhaseIDToDags = std::unordered_map<PhaseShiftIdType, std::vector<const DAG*>>;
                std::unordered_map< DAGHashType, PhaseIDToDags > other_hashtags;
                for (const DAG* const dag : dags) { //
                    for (const auto& [other_hash, phaseShiftId]
                            : dag->other_hash_values_with_phase_shift_id()) {
                        if (phaseShiftId == kNoPhaseShift)
                            other_hashtags[other_hash][phaseShiftId] = std::vector<const DAG*>();
                        else
                            other_hashtags[other_hash][phaseShiftId].push_back(dag);
                    }
                } // end for dags

                /*
                 * build dags_to_verify
                 * dags_to_verify[0]: representative of another ecc
                 * dags_to_verify[1]: EquivClassTag of the other ecc
                 * dags_to_verify[2]: dags to verify in { phase_id: [dag_x, ...] } of this ecc
                 */
                std::vector<
                        std::tuple<const DAG*, EquivClassTag, PhaseIDToDags>
                > dags_to_verify;
                for (auto& [other_hash, phaseIDToDags] : other_hashtags) {
//                for (auto it = other_hashtags.begin(); it != other_hashtags.end(); it++) {
//                    const DAGHashType other_hash = it->first;
//                    PhaseIDToDags& phaseIDToDags = it->second;
                    // try to find other_hash in dataset
                    // if not found, no need to consider equivalence of dags in phaseIDToDags
                    if (dataset.find(other_hash) != dataset.end()) {
                        // find the ecc having other_hash classed and class_hashes
                        size_t loc = 0;
                        for (loc = 0; loc < class_hashes.size(); loc++) {
                            if (class_hashes[loc].front().first == other_hash)
                                break;
                        }
                        assert(loc < class_hashes.size()); // must be found
                        for (size_t k = 0; k < class_hashes[loc].size(); k++) {
                            const EquivClassTag& other_hashtag = class_hashes[loc][k];
                            const DAG* rep_dag = classes[loc][k]->get_representative();
                            dags_to_verify.emplace_back(rep_dag, other_hashtag, std::move(phaseIDToDags));
                            assert(!std::get<2>(dags_to_verify.back()).empty()); // ATTENTION Colin : clang-tidy
                        }
                    }
                } // end for other_hashtags

                // verify equivalence in dags_to_verify
                for (const auto& [other_rep_dag, other_ecctag, phaseIDToDags] : dags_to_verify) {
                    bool equivalence_found = false;
                    for (const auto& it : phaseIDToDags) {
                        if (it.first == kNoPhaseShift) {
                            equivalence_found = false;
                            /* TODO Colin : equivalent(ecc->get_representative(), other_rep_dag, parameters_for_fingerprint) */
                            break;
                        }
                    }
                    // only need to find one equivalence
                    if (!equivalence_found) for (const auto& [phase_id, possible_dags] : phaseIDToDags) {
                        if (phase_id == kNoPhaseShift)
                            continue;
                        bool input_param_tried = false;
                        for (const DAG* dag : possible_dags) {
                            const int dag_n_input_params = dag->get_num_input_parameters();
                            const int dag_n_tot_params = dag->get_num_total_parameters();
                            const bool fixed_for_all_dags =
                                (0 <= phase_id && phase_id < dag_n_input_params) ||
                                (dag_n_tot_params <= phase_id && phase_id < dag_n_tot_params + dag_n_input_params) ||
                                (kCheckPhaseShiftOfPiOver4Index < phase_id && phase_id < kCheckPhaseShiftOfPiOver4Index + 8);
                            if (fixed_for_all_dags) {
                                if (input_param_tried)
                                    continue;
                                else
                                    input_param_tried = true;
                            }
                            if (false /*TODO Colin : equivalent(dag, other_rep_dag, parameters_for_fingerprint, phase_id) */) {
                                equivalence_found = true;
                                break;
                            }
                        } // end for possible_dags
                        if (equivalence_found)
                            break;
                    } // end (if) for phaseIDToDags
                    if (equivalence_found) {
                        // store found equivalence info (other_ecctag, hashtag)
                        equiv_edges[hashtag].emplace_back(other_ecctag);
                        equiv_edges[other_ecctag].emplace_back(hashtag);
                    }
                } // end for dags_to_verify
            // iterate over each ecc
            }
        } // end iterate over each ecc


    }

} // namespace quartz
