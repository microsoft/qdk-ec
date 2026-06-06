import os
from collections.abc import Mapping
import hjson
from errata.circuits.clifford import Operand
from errata.simulation.noise import LocationStochasticNoiseModel, Probability, FaultSet
from errata.simulation.fault_evaluation import Location
from errata.statistics.distribution import (
    DiscreteDistribution,
    ProductDistribution,
    ExplicitDiscreteDistribution,
)


this_dir = os.path.abspath(os.path.dirname(__file__))
import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb
from google.protobuf.json_format import MessageToJson, MessageToDict


def visualize_noise(
    noise: LocationStochasticNoiseModel[Probability],
    location: Location,
    support_indices: Mapping[Operand, int],
) -> list[vis_pb.NoiseDistribution]:
    product_distribution = noise.faults_at(location)
    pmf_distributions = expand_product_distributions(product_distribution)
    return [
        noise_distribution_of(distribution, location.level, support_indices)
        for distribution in pmf_distributions
    ]


def noise_distribution_of(
    distribution: ExplicitDiscreteDistribution[FaultSet, Probability],
    expected_level: int,
    support_indices: Mapping[Operand, int],
) -> vis_pb.NoiseDistribution:
    masses: list[vis_pb.NoiseMass] = []
    for fault_set, probability in distribution.masses:
        faults: list[vis_pb.Fault] = []
        for fault in fault_set:
            assert (
                fault.level == expected_level
            ), f"Unexpected fault level from {fault}, expected {expected_level}"
            faults.append(
                vis_pb.Fault(type=fault.type, qubit=support_indices[fault.qubit])
            )
        masses.append(vis_pb.NoiseMass(faults=faults, probability=probability))
    return vis_pb.NoiseDistribution(masses=masses)


def expand_product_distributions(
    distribution: DiscreteDistribution[FaultSet, Probability],
) -> list[ExplicitDiscreteDistribution[FaultSet, Probability]]:
    if isinstance(distribution, ExplicitDiscreteDistribution):
        return [distribution]
    if isinstance(distribution, ProductDistribution):
        results = []
        for marginal in distribution.marginals:
            results += expand_product_distributions(marginal)
        return results
    raise ValueError("Unsupported distribution type", type(distribution))


def generate_example(
    library: pb2.Library,
    basename: str,
    generate_json: bool = False,
    generate_hjson: bool = False,
    generate_pb: bool = True,
    generate_pb_text: bool = True,
) -> None:
    # write to a json file; note that the generated json cannot be used directly
    # in the frontend: it is incompatible with `protobuf-ts` data structures
    if generate_json:
        with open(os.path.join(this_dir, f"{basename}.json"), "w") as f:
            json_str = MessageToJson(
                library,
                use_integers_for_enums=False,
                ensure_ascii=False,
                # indent=4,
                indent=None,
            )
            f.write(json_str)

    # write to an hjson file
    if generate_hjson:
        with open(os.path.join(this_dir, f"{basename}.hjson"), "w") as f:
            hjson.dump(MessageToDict(library), f, ensure_ascii=False)

    # write to the binary wired form
    if generate_pb:
        with open(os.path.join(this_dir, f"{basename}.deq.bin"), "wb") as f:
            f.write(library.SerializeToString())

    # generate human readable protobuf
    if generate_pb_text:
        with open(
            os.path.join(this_dir, f"{basename}.deq.bin.txt"), "w", encoding="utf8"
        ) as f:
            f.write(str(library))
