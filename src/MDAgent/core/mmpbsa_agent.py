from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
import inspect
import logging
from molecular_simulations.simulate.mmpbsa import MMPBSA
import parsl
from parsl import python_app, Config
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Parsl apps that directly execute the simulation tasks
# These run in separate processes/threads and don't use the Agent framework
@python_app
def parsl_mmpbsa(fe_kwargs: dict[str, Any]) -> float:
    """Execute MMPBSA calculation directly in Parsl worker."""
    from molecular_simulations.simulate.mmpbsa import MMPBSA
    
    path = fe_kwargs['dcd'].parent

    try:
        mmpbsa = MMPBSA(**fe_kwargs)
        mmpbsa.run()
        fe = mmpbsa.free_energy
    except Exception as e:
        logger.warn(f'MMPBSA for {path} failed: {e}')
        fe = None
    
    return {'path': path, 'success': fe is not None, 'fe': fe}

class FreeEnergy(Agent):
    """Agent wrapper for free energy - not used in Parsl execution."""
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
        # This method is not used in Parsl mode
        # Parsl uses parsl_mmpbsa directly
        mmpbsa = self.calculator(**fe_kwargs)
        return mmpbsa.run()


class FECoordinator(Agent):
    """Coordinator that orchestrates Parsl tasks."""
    def __init__(
        self, 
        parsl_config: Config,
    ) -> None:
        super().__init__()
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
    async def run_mmpbsa(
        self,
        fe_kwargss: list[dict[str, Any]]
    ) -> list[float]:
        """Submit MMPBSA tasks to Parsl and wait for completion."""
        futures = []
        for fe_kwargs in fe_kwargss:
            # Call the Parsl app directly
            app_future = parsl_mmpbsa(fe_kwargs)
            futures.append(asyncio.wrap_future(app_future))

        return await asyncio.gather(*futures)

    @action
    async def deploy(
        self,
        paths: list[Path],
        fe_kwargss: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deploy full MD workflow using Parsl."""
        logger.info('Computing free energy with MM-PBSA!')
        futures = []
        # Update fe_kwargs with paths from simulations
        for path, fe_kwargs in zip(paths, fe_kwargss):
            logger.info(f'Will run at: {path}')
            fe_kwargs.update({
                'top': path / 'system.prmtop',
                'dcd': path / 'prod.dcd'
            })

            futures.append(asyncio.wrap_future(parsl_mmpbsa(fe_kwargs)))

        return await asyncio.gather(*futures)
