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

# Parsl apps that directly execute the simulation tasks
# These run in separate processes/threads and don't use the Agent framework
@python_app
def parsl_build(path: Path, pdb: Path, build_kwargs: dict[str, Any]) -> Path:
    """Execute build task directly in Parsl worker."""
    from molecular_simulations.build import ImplicitSolvent, ExplicitSolvent
    
    solvent = build_kwargs.get('solvent', 'implicit')
    
    if solvent == 'implicit':
        builder = ImplicitSolvent(path=path, pdb=pdb, 
                                  **build_kwargs)
    else:  # explicit
        builder = ExplicitSolvent(path=path, pdb=pdb,
                                  **build_kwargs)
    
    builder.build()
    return builder.out.parent

@python_app
def parsl_simulate(path: Path, sim_kwargs: dict[str, Any]) -> Path:
    """Execute simulation task directly in Parsl worker."""
    from molecular_simulations.simulate import ImplicitSimulator, Simulator
    
    solvent = sim_kwargs.get('solvent', 'implicit')
    
    if solvent == 'implicit':
        simulator = ImplicitSimulator(path, 
                                      **sim_kwargs)
    else:  # explicit
        simulator = Simulator(path,
                              **sim_kwargs)
    
    simulator.run()
    return simulator.path

@python_app
def parsl_mmpbsa(fe_kwargs: dict[str, Any]) -> float:
    """Execute MMPBSA calculation directly in Parsl worker."""
    from molecular_simulations.simulate.mmpbsa import MMPBSA
    
    mmpbsa = MMPBSA(**fe_kwargs)
    return mmpbsa.run()

def appfuture_to_async(app_fut) -> asyncio.Future:
    """Convert a Parsl AppFuture to an asyncio Future."""
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
    """Agent wrapper for building - not used in Parsl execution."""
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
        # This method is not used in Parsl mode
        # Parsl uses parsl_build directly
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
    """Agent wrapper for simulation - not used in Parsl execution."""
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
        # This method is not used in Parsl mode
        # Parsl uses parsl_simulate directly
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

class MDCoordinator(Agent):
    """Coordinator that orchestrates Parsl tasks."""
    def __init__(
        self, 
        builder: Handle[Builder], 
        simulator: Handle[MDSimulator],
        parsl_config: Config,
    ) -> None:
        super().__init__()
        self.builder = builder
        self.simulator = simulator
        self.config = parsl_config
        self.dfk = None

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
        """Submit build tasks to Parsl and wait for completion."""
        futures = []
        for path, pdb, build_kwargs in zip(paths, structural_inputs, build_kwargss):
            # Call the Parsl app directly, not through the Agent
            app_future = parsl_build(path, pdb, build_kwargs)
            futures.append(app_future)

        # Convert Parsl futures to async futures and wait
        async_futures = [asyncio.wrap_future(f) for f in futures]
        return await asyncio.gather(*async_futures)

    @action
    async def run_simulation(
        self,
        simulation_paths: list[Path],
        sim_kwargss: list[dict[str, Any]]
    ) -> list[Path]:
        """Submit simulation tasks to Parsl and wait for completion."""
        futures = []
        for path, sim_kwargs in zip(simulation_paths, sim_kwargss):
            # Call the Parsl app directly
            app_future = parsl_simulate(path, sim_kwargs)
            futures.append(app_future)
        
        # Convert and wait
        async_futures = [asyncio.wrap_future(f) for f in futures]
        return await asyncio.gather(*async_futures)

    @action
    async def deploy_md(
        self,
        paths: list[Path],
        initial_pdbs: list[Path],
        build_kwargss: list[dict[str, Any]],
        sim_kwargss: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deploy full MD workflow using Parsl."""
        logger.info(f'Building systems at: {paths}')
        built_paths = await self.build_system(paths, initial_pdbs, build_kwargss)
        
        logger.info(f'Successfully built systems. Simulating at: {built_paths}')
        sim_paths = await self.run_simulation(built_paths, sim_kwargss)

        return [{'build': path, 'sim': 'success'} for path in paths]
