from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
import inspect
import logging
from molecular_simulations.build import ImplicitSolvent, ExplicitSolvent
from molecular_simulations.simulate import ImplicitSimulator, Simulator
from molecular_simulations.simulate.mmpbsa import MMPBSA
import parsl
from parsl import python_app, Config
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
    
@python_app
def deploy_task(call: Callable,
                args: list[Any]=[]) -> Any:
    future = call(*args)
    return future

@python_app
def deploy_build(builder: Agent,
                 args: list[Any]):
    builder.build(*args)

@python_app
def deploy_sim(sim: Agent,
               args: list[Any]):
    sim.simulate(*args)

@python_app
def deploy_mmpbsa(mmpbsa: Agent,
                  args: list[Any]):
    mmpbsa.measure(*args)

def appfuture_to_async(app_fut) -> asyncio.Future:
    loop = asyncio.get_running_loop()
    afut = loop.create_future()

    def _done(parsl_fut):
        try:
            res = parsl_fut.result()
            loop.call_soon_threadsafe(afut.set_result, res)
        except Exception as e:
            loop.call_soon_threadsafe(afut.set_exception, e)

    app_fut.add_done_callback(_done)
    return afut

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
        pdb: Path,
        build_kwargs: dict[str, Any],
    ) -> Path:
        solvent = build_kwargs.get('solvent', 'implicit')

        match solvent:
            case 'implicit':
                keys = set(inspect.signature(self.implicit).parameters)
                kwargs = {k: build_kwargs[k] for k in build_kwargs if k in keys}

                builder = self.implicit(path=path, pdb=pdb, **kwargs)

            case 'explicit':
                keys = set(inspect.signature(self.explicit).parameters)
                kwargs = {k: build_kwargs[k] for k in build_kwargs if k in keys}

                builder = self.explicit(path=path, pdb=pdb, **kwargs)

            case _:
                raise ValueError('Something terrible happened!')

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

        match solvent:
            case 'implicit':
                keys = set(inspect.signature(self.implicit).parameters)
                kwargs = {k: sim_kwargs[k] for k in sim_kwargs if k in keys}

                simulator = self.implicit(path, **kwargs)
            
            case 'explicit':
                keys = set(inspect.signature(self.explicit).parameters)
                kwargs = {k: sim_kwargs[k] for k in sim_kwargs if k in keys}

                simulator = self.explicit(path, **kwargs)
            
            case _:
                raise ValueError('Something terrible happened!')

        simulator.run()

        return simulator.path

class FreeEnergy(Agent):
    def __init__(
        self,
        calculator: object=MMPBSA
    ) -> None:
        self.calculator = calculator

    @action
    async def measure(
        self,
        fe_kwargs: dict[str, Any],
    ) -> float:
        mmpbsa = MMPBSA(**fe_kwargs)

        return mmpbsa.run()


class MDCoordinator(Agent):
    def __init__(
        self, 
        builder: Handle[Builder], 
        simulator: Handle[MDSimulator],
        free_energy: Handle[FreeEnergy],
        parsl_config: Config,
    ) -> None:
        super().__init__()
        self.builder = builder
        self.simulator = simulator
        self.free_energy = free_energy

        self.config = parsl_config

    async def agent_on_startup(self) -> None:
        """Initialize Parsl on agent startup."""
        logger.info(f'Initializing Parsl workers')
        self.dfk = parsl.load(self.config)

    async def agent_on_shutdown(self) -> None:
        """Clean up Parsl on agent shutdown."""
        logger.info('Cleaning up Parsl')
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()


    @action
    async def build_system(
        self,
        paths: list[Path],
        structural_inputs: list[Path],
        build_kwargss: list[dict[str, Any]]
    ) -> list[Path]:
        futures = []
        for args in zip(paths, structural_inputs, build_kwargss):
            futures.append(asyncio.wrap_future(deploy_build(self.builder, args)))

        async_futures = [appfuture_to_async(f) for f in futures]
        return await asyncio.gather(*async_futures)

    @action
    async def run_simulation(
        self,
        simulation_paths: list[Path],
        sim_kwargss: list[dict[str, Any]]
    ) -> Path:
        futures = []
        for args in zip(simulation_paths, sim_kwargss):
            futures.append(asyncio.wrap_future(deploy_sim(self.simulator, args)))
        
        async_futures = [appfuture_to_async(f) for f in futures]
        return await asyncio.gather(*futures)

    @action
    async def run_mmpbsa(
        self,
        fe_kwargss: list[dict[str, Any]]
    ) -> float:
        futures = [asyncio.wrap_future(deploy_mmpbsa(self.free_energy, fe_kwargs)) for fe_kwargs in fe_kwargss]
        async_futures = [appfuture_to_async(f) for f in futures]

        return await asyncio.gather(*futures)

    @action
    async def deploy_md(
        self,
        paths: list[Path],
        initial_pdbs: list[Path],
        build_kwargss: list[dict[str, Any]],
        sim_kwargss: list[dict[str, Any]],
        fe_kwargss: list[dict[str, Any]]
    ) -> dict[str, Any]:
        logger.info(f'Building systems at: {paths}')
        await self.build_system(paths, initial_pdbs, build_kwargss)
        
        logger.info(f'Successfully built system. Simulating!')
        await self.run_simulation(paths, sim_kwargss)

        logger.info('Computing free energy with MM-PBSA!')
        for path, fe_kwargs in zip(paths, fe_kwargss):
            fe_kwargs.update({
                'top': path / 'system.prmtop',
                'dcd': path / 'prod.dcd'
            })

        await self.run_mmpbsa(fe_kwargss)

        return [{'build': path,
                 'sim': 'success'} for path in paths]
