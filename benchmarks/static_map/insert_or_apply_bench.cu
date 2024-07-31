/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <benchmark_defaults.hpp>
#include <benchmark_utils.hpp>

#include <cuco/static_map.cuh>
#include <cuco/utility/key_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <thrust/shuffle.h>
#include <thrust/random.h>


using namespace cuco::benchmark;
using namespace cuco::utility;

struct plus_op {
    template <typename T, cuda::thread_scope Scope>
    __device__ void operator()(cuda::atomic_ref<T, Scope> ref, T val)
    {
        ref.fetch_add(val, cuda::memory_order_relaxed);
    }
};

/**
* @brief A benchmark evaluating `cuco::static_map::insert_or_apply` performance
*/
template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> static_map_insert_or_apply(
nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
    using pair_type = cuco::pair<Key, Value>;

    auto const num_keys     = state.get_int64_or_default("NumInputs", defaults::N);
    auto const occupancy    = state.get_float64_or_default("Occupancy", defaults::OCCUPANCY);
    auto const cardinality = state.get_int64_or_default("Cardinality", defaults::N);

    if (num_keys < cardinality)
    {
        state.skip("num_keys < cardinality");
        return;
    }

    auto pairs_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<Key>(0),
        cuda::proclaim_return_type<cuco::pair<Key, Value>>([cardinality] __device__(auto i) {
        return cuco::pair<Key, Value>{i % cardinality, 1};
        }));

    thrust::device_vector<pair_type> pairs(pairs_begin, pairs_begin + num_keys);
    thrust::default_random_engine g;
    thrust::shuffle(pairs.begin(), pairs.end(), g);

    thrust::device_vector<Key> output_keys(num_keys);
    thrust::device_vector<Value> output_values(num_keys);

    state.add_element_count(num_keys);

    plus_op plusop{};
    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {

        timer.start();

        cuco::static_map map{num_keys, occupancy, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{0}, {}, cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};

        map.insert_or_apply_async(pairs.begin(), pairs.end(), plusop, {launch.get_stream()});
        map.retrieve_all(output_keys.begin(), output_values.begin(), {launch.get_stream()});

        timer.stop();
    });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> static_map_insert_or_apply(
nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(static_map_insert_or_apply,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                    defaults::VALUE_TYPE_RANGE,
                                    nvbench::type_list<distribution::uniform>))
.set_name("static_map_insert_or_apply_uniform_multiplicity")
.set_type_axes_names({"Key", "Value", "Distribution"})
.set_max_noise(defaults::MAX_NOISE)
.add_int64_axis("Cardinality", {1, 10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
.add_int64_axis("NumInputs", {1, 10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000});

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) == sizeof(Value)), void> thrust_insert_or_apply(
nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
    auto const num_keys     = state.get_int64_or_default("NumInputs", defaults::N);
    auto const cardinality = state.get_int64_or_default("Cardinality", defaults::N);

    if (num_keys < cardinality)
    {
        state.skip("num_keys < cardinality");
        return;
    }

    auto keys_begin = thrust::make_transform_iterator(
        thrust::counting_iterator<Key>(0),
        cuda::proclaim_return_type<Key>([cardinality] __device__(auto i) {
        return static_cast<Key>(i % cardinality);
        }));

    thrust::device_vector<Key> keys(keys_begin, keys_begin + num_keys);
    thrust::default_random_engine g;
    thrust::shuffle(keys.begin(), keys.end(), g);

    thrust::device_vector<Value> values(num_keys, 1);

    thrust::device_vector<Key> output_keys(num_keys);
    thrust::device_vector<Value> output_values(num_keys);

    state.add_element_count(num_keys);

    state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        timer.start();
            thrust::sort_by_key(thrust::cuda::par.on(launch.get_stream()), keys.begin(), keys.end(), values.begin(), thrust::greater<int>());
            thrust::reduce_by_key(thrust::cuda::par.on(launch.get_stream()), keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());
        timer.stop();
    });
}

template <typename Key, typename Value, typename Dist>
std::enable_if_t<(sizeof(Key) != sizeof(Value)), void> thrust_insert_or_apply(
nvbench::state& state, nvbench::type_list<Key, Value, Dist>)
{
state.skip("Key should be the same type as Value.");
}

NVBENCH_BENCH_TYPES(thrust_insert_or_apply,
                    NVBENCH_TYPE_AXES(defaults::KEY_TYPE_RANGE,
                                    defaults::VALUE_TYPE_RANGE,
                                    nvbench::type_list<distribution::uniform>))
.set_name("thrust_insert_or_apply_uniform_multiplicity")
.set_type_axes_names({"Key", "Value", "Distribution"})
.set_max_noise(defaults::MAX_NOISE)
.add_int64_axis("Cardinality", {1, 10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
.add_int64_axis("NumInputs", {1, 10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000});