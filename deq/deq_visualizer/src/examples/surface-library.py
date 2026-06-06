from pathlib import Path
from errata.gadgetry.gadget import Gadget, Channel
from errata.codes import StabilizerCode
from errata.circuits.surface_code_catalog import (
    make_surface_code_syndrome_extraction,
    CircuitType,
)
from errata.circuits.circuit import Location
from errata.circuits.clifford import Measurement
import errata.circuits.circuit as circuit
from errata.circuits import clifford
from errata.simulation.stabilizer_evaluation import stabilizer_group_of
from errata.pauli import Pauli
from errata.simulation.noise import UniformNoiseModel
from errata.simulation.noise import LocationStochasticNoiseModel, Probability
import common
from deq.visual.adaptor import visualize_operation
import deq.proto.deq_bin_pb2 as pb2
import deq.proto.visualizer_pb2 as vis_pb

DEFAULT_RADIUS = 0.1


def main(d: int = 3):
    factory = SurfaceCodeFactory(d)
    init = factory.prepare_z()
    idle = factory.idle()
    meas = factory.measure_z()

    data_support = make_consistent_support(d, no_ancilla=True)
    all_support = make_consistent_support(d, no_ancilla=False)

    init_realization = visualize_realization_of(
        d, init.realization.locations, all_support
    )
    idle_realization = visualize_realization_of(
        d,
        idle.realization.locations,
        all_support,
        noise=UniformNoiseModel(1e-3),
    )
    meas_realization = visualize_realization_of(
        d, meas.realization.locations, data_support, block_height=0.1
    )
    scale = scale_of(d)

    library = pb2.Library(
        gadget_types=[
            pb2.GadgetType(
                gtype=1,
                name="🆕|0⟩",
                description="Initialize |0⟩",
                outputs=[
                    pb2.GadgetType.Port(
                        ptype=1, tag="output", relative=vis_pb.Position(t=1.02)
                    )
                ],
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[1, 1, 1]),
                        material=vis_pb.Material(type="transmission", color="#3a20ff"),
                        relative=vis_pb.Position(t=0.5),
                    )
                ],
                realization=init_realization,
            ),
            pb2.GadgetType(
                gtype=2,
                name="⌛",
                description="Idle Logical Cycle",
                inputs=[
                    pb2.GadgetType.Port(
                        ptype=1, tag="input", relative=vis_pb.Position(t=-0.02)
                    )
                ],
                outputs=[
                    pb2.GadgetType.Port(
                        ptype=1, tag="output", relative=vis_pb.Position(t=1.02)
                    )
                ],
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[1, 1, 1]),
                        material=vis_pb.Material(type="transmission", color="#FF8F20"),
                        relative=vis_pb.Position(t=0.5),
                    )
                ],
                realization=idle_realization,
            ),
            pb2.GadgetType(
                gtype=3,
                name="👀Ẑ",
                description="Measure Z logical readout",
                inputs=[
                    pb2.GadgetType.Port(
                        ptype=1, tag="input", relative=vis_pb.Position(t=-0.025)
                    )
                ],
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[0.05, 1, 1]),
                        material=vis_pb.Material(type="transmission", color="#3a80aa"),
                        relative=vis_pb.Position(t=0.025),
                    )
                ],
                realization=meas_realization,
            ),
        ],
        port_types=[
            pb2.PortType(
                ptype=1,
                name="▩rsc",
                description="rotated surface code",
                mesh=[
                    vis_pb.Mesh(
                        geometry=vis_pb.Geometry(type="box", size=[0.02, 0.8, 0.8]),
                        material=vis_pb.Material(type="standard", color="#00FFFF"),
                        relative=vis_pb.Position(t=0.0),
                    )
                ],
                positions=[
                    vis_pb.Position(
                        i=(i - (d - 1) / 2) * scale, j=(j - (d - 1) / 2) * scale
                    )
                    for i, j in data_support
                ],
            )
        ],
        program=[
            pb2.Instruction(gadget=pb2.Gadget(gtype=1, position=vis_pb.Position(t=0))),
            pb2.Instruction(
                gadget=pb2.Gadget(
                    gtype=2,
                    connectors=[pb2.Gadget.Connector(gid=1, port=0)],
                    position=vis_pb.Position(t=1.2),
                )
            ),
            pb2.Instruction(
                gadget=pb2.Gadget(
                    gtype=3,
                    connectors=[pb2.Gadget.Connector(gid=2, port=0)],
                    position=vis_pb.Position(t=2.4),
                )
            ),
        ],
        visual_config=vis_pb.VisualConfig(qubit_radius=scale * DEFAULT_RADIUS),
    )

    common.generate_example(library, Path(__file__).stem)


class SurfaceCodeFactory:
    def __init__(self, distance: int):
        self.distance = distance
        syndrome = make_surface_code_syndrome_extraction(
            distance, CircuitType.cnot_based, x_distance=distance, z_distance=distance
        )
        self.ancillas = circuit.output_support_of(syndrome) - circuit.input_support_of(
            syndrome
        )
        discard = circuit.scheduled_concurrently(map(clifford.Discard, self.ancillas))
        self.syndrome = circuit.concatenate(syndrome, discard)

    @property
    def data(self):
        return circuit.output_support_of(self.syndrome) - self.ancillas

    @property
    def code(self):
        return StabilizerCode(abs(stabilizer_group_of(self.syndrome)).generators)

    def prepare_z(self):
        trivial_code = StabilizerCode([])
        prep_z = circuit.scheduled_concurrently(
            map(clifford.PrepareZero, circuit.input_support_of(self.syndrome))
        )
        prep_z = circuit.concatenate(prep_z, self.syndrome)
        realization = Channel(
            locations=prep_z, code_in=trivial_code, code_out=self.code
        )
        objective = Channel(
            locations=circuit.scheduled_concurrently([clifford.PrepareZero(0)])
        )
        return Gadget(realization=realization, objective=objective)

    def idle(self):
        realization = Channel(self.syndrome, code_in=self.code, code_out=self.code)
        objective = Channel(locations=[], code_in=self.code, code_out=self.code)
        return Gadget(realization=realization, objective=objective)

    def measure_z(self):
        data = circuit.output_support_of(self.syndrome)
        discard = circuit.scheduled_concurrently(map(clifford.Discard, data))
        measure_z = circuit.scheduled_concurrently(
            [
                Measurement(Pauli({index: "Z"}), correction=Pauli({index: "X"}))
                for index in circuit.input_support_of(self.syndrome)
            ]
        )
        measure_z = circuit.concatenate(measure_z, discard)
        trivial_code = StabilizerCode([])
        realization = Channel(measure_z, code_in=self.code, code_out=trivial_code)
        objective = Channel(
            locations=circuit.scheduled_consecutively(
                [clifford.Measurement.from_string("Z"), clifford.Discard(0)]
            ),
        )
        return Gadget(realization=realization, objective=objective)


def make_consistent_support(d: int, no_ancilla: bool) -> list[tuple[float, float]]:
    """
    Make consistent physical qubit (data + ancilla) numbering
    data qubit goes first, followed by ancilla qubits
    """
    support: list[tuple[float, float]] = []
    # add data qubits
    for i in range(d):
        for j in range(d):
            support.append((i, j))
    if no_ancilla:
        return support
    # add ancilla qubits
    for i in range(d + 1):
        for j in range(d + 1):
            if i == 0:
                if j % 2 == 1 and j > 0 and j < d:
                    support.append((i - 0.5, j - 0.5))
            elif i == d:
                if j % 2 == 0 and j > 0 and j < d:
                    support.append((i - 0.5, j - 0.5))
            elif j == 0:
                if i % 2 == 0:
                    support.append((i - 0.5, j - 0.5))
            elif j == d:
                if i % 2 == 1:
                    support.append((i - 0.5, j - 0.5))
            else:
                support.append((i - 0.5, j - 0.5))
    return support


def support_to_scaled_coordinates(
    d: int, coordinate: tuple[float, float]
) -> vis_pb.Position2D:
    x, y = coordinate
    center = (d - 1) / 2
    dx, dy = x - center, y - center
    # we want to fit d+1 columns into a block of 0.8
    scale = scale_of(d)
    return vis_pb.Position2D(i=dx * scale, j=dy * scale)


def scale_of(d: int) -> float:
    return 0.8 / (d + 1)


def visualize_realization_of(
    d: int,
    locations: list[Location],
    supports: list[tuple[float, float]],
    block_height: float = 1.0,
    noise: LocationStochasticNoiseModel[Probability] | None = None,
) -> vis_pb.Realization:
    # first calculate the range of levels
    levels = [location.level for location in locations]
    min_level = min(levels) if levels else 0
    max_level = max(levels) if levels else 0
    avr_level = (min_level + max_level) / 2
    t_scale = 1
    if max_level > min_level:
        t_scale = 0.6 / (max_level - min_level)

    def t_from(level: int) -> float:
        return (level - avr_level) * t_scale * block_height + block_height / 2

    # then create the realization
    support_indices = {support: i for i, support in enumerate(supports)}
    return vis_pb.Realization(
        locations=[
            vis_pb.Location(
                t=t_from(location.level),
                operation=visualize_operation(location.operation, support_indices),
                noises=(
                    None
                    if noise is None
                    else common.visualize_noise(noise, location, support_indices)
                ),
            )
            for location in locations
        ],
        positions=[support_to_scaled_coordinates(d, support) for support in supports],
    )


if __name__ == "__main__":
    main()
