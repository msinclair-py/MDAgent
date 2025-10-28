from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
import inspect
import logging
from molecular_simulations.build import ImplicitSolvent, ExplicitSolvent
from molecular_simulations.simulate import ImplicitSimulator, Simulator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class Builder(Agent):
    def __init__(
        self,
        implicit: object=ImplicitSolvent,
        explicit: object=ExplicitSolvent
    ) -> None:
        self.implicit = implicit
        self.explicit = explicit

    @action
    async def build(
        self,
        path: Path,
        build_kwargs: dict[str, Any],
    ) -> Path:
        solvent = build_kwargs.get('solvent', 'implicit')

        if solvent == 'implicit':
            keys = set(inspect.signature(self.implicit).parameters)
            kwargs = {k: build_kwargs[k] for k in in build_kwargs if k in keys}

            builder = self.implicit(path, **kwargs)
        else:
            keys = set(inspect.signature(self.explicit).parameters)
            kwargs = {k: build_kwargs[k] for k in in build_kwargs if k in keys}

            builder = self.explicit(path, **build_kwargs)

        builder.build()

        return builder.out.parent

class MDSimulator(Agent):
    def __init__(
        self,
        implicit: object=ImplicitSimulator,
        explicit: object=Simulator
    ) -> None:
        self.implicit = implicit
        self.explicit = explicit

    @action
    async def simulate(
        self,
        path: Path,
        sim_kwargs: dict[str, Any],
    ) -> Path:
        solvent = sim_kwargs.get('solvent', 'implicit')
        if solvent == 'implicit':
            keys = set(inspect.signature(self.implicit).parameters)
            kwargs = {k: sim_kwargs[k] for k in in sim_kwargs if k in keys}

            simulator = self.implicit(path, **sim_kwargs)
        else:
            keys = set(inspect.signature(self.explicit).parameters)
            kwargs = {k: sim_kwargs[k] for k in in sim_kwargs if k in keys}

            simulator = self.explicit(path, **sim_kwargs)

        simulator.run()

        return simulator.path

class MDCoordinator(Agent):
    def __init__(
        self, 
        builder: Handle[Builder], 
        simulator: Handle[MDSimulator]
    ) -> None:
        super().__init__()
        self.builder = builder
        self.simulator = simulator

    @action
    async def build_system(
        self,
        structural_input: Path,
        build_kwargs: dict[str, Any]
    ) -> Path:
        return self.builder.build(structural_input, build_kwargs)

    @action
    async def run_simulation(
        self,
        simulation_path: Path,
        sim_kwargs: dict[str, Any]
    ) -> Path:
        return self.simulator.simulate(simulation_path, sim_kwargs)

    @action
    async def deploy_md(
        self,
        initial_pdb: Path,
        build_kwargs: dict[str, Any],
        sim_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        path = await self.build_system(initial_pdb, build_kwargs)
        await self.run_simulation(path, sim_kwargs)

        return {'build': path,
                'sim': 'success'}
