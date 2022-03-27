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
        std::vector< std::vector<std::pair<DAGHashType, unsigned>> > class_hashes(dataset.size());
        std::vector< std::vector<std::unique_ptr<EquivalenceClass>> > classes(dataset.size());
        // convert unordered_map obj dataset to vector to enable parallelization
        std::vector<std::pair< DAGHashType, std::vector<std::unique_ptr<DAG>>* >> vec_dataset;
        for (auto& [hash, dags] : dataset) {
            vec_dataset.emplace_back(hash, &dags);
        }

        #pragma omp parallel for default(none) shared(vec_dataset, class_hashes, classes)
        for (size_t i = 0; i < vec_dataset.size(); i++) {
            // build vector<EquivalenceClass> at classes[i]
            DAGHashType hashtag = vec_dataset[i].first;
            std::vector<std::unique_ptr<DAG>>* dags = vec_dataset[i].second;
            // more than one dag, need to check if they are really equivalent
            std::vector<std::pair<DAGHashType, unsigned>>& this_class_hashes = class_hashes[i];
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
                        if (true /* TODO Colin : equivalent(dag, other_dag) */) {
                            found_equivalent_one = true;
                            // ATTENTION : copy construct a dag
                            inserted_ecc->insert(std::make_unique<DAG>(*dag));
                            break;
                        }
                    } // for this_eccs
                    if (!found_equivalent_one) {
                        // non-equivalent dag with the same hash, build a new ecc
                        insert_ecc_with_dag(dag);
                    }
                }
            } // for dags
        } // for vec_dataset (paralleled)


    }

} // namespace quartz
