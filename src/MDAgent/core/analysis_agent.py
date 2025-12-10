from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
import inspect
import logging
import numpy as np
import parsl
from parsl import python_app, Config
from pathlib import Path
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

@parsl.python_app
def parsl_energy(mode: Literal['static', 'dynamic', 'fingerprint']='static',
                 **kwargs) -> np.ndarray:
    from molecular_simulations.analysis import (StaticInteractionEnergy,
                                                DynamicInteractionEnergy,
                                                Fingerprinter)
    energy_calculation = {'static': StaticInteractionEnergy,
                          'dynamic': DynamicInteractionEnergy,
                          'fingerprint': Fingerprinter}
    calculation = energy_calculation[mode](**kwargs)

    match mode:
        case 'static':
            calculation.compute()
            return calculation.interactions
        case 'dynamic':
            calculation.compute_energies()
            return calculation.energies
        case 'fingerprint':
            calculation.run()
            return calculation.binder_fingerprint

@parsl.python_app
def parsl_cluster(path: Path,
                  pattern: str='',
                  max_clusters: int=10,
                  stride: int=1,) -> tuple[np.ndarray, np.ndarray]:
    from molecular_simulations.analysis.autocluster import AutoKMeans
    autocluster = AutoKMeans(data_directory=path, 
                             pattern=pattern, 
                             max_clusters=max_clusters,
                             stride=stride)
    autocluster.run()

    return autocluster.labels, autocluster.centers

@parsl.python_app
def parsl_ppi(top: Path,
              traj: Path,
              out: Path,
              sel1: str='chainID A',
              sel2: str='chainID B',
              **kwargs) -> dict[str, dict[str, float]]:
    from molecular_simulations.analysis.cov_ppi import PPInteractions
    ppi = PPInteractions(top, traj, out, sel1, sel2, **kwargs)
    ppi.run()
    
    return ppi.results


class AnalysisCoordinator(Agent):
    def __init__(
        self,
        parsl_config: Config,
    ) -> None:
        super().__init__()
        self.parsl_config = parsl_config
        self.dfk = None

    async def agent_on_startup(self) -> None:
        logger.info(f'Initializing Parsl workers')
        self.dfk = parsl.load(self.config)

    async def agent_on_shutdown(self) -> None:
        logger.info('Cleaning up Parsl')
        if self.dfk:
            self.dfk.cleanup()
            self.dfk = None
        parsl.clear()

    @action
    async def compute_energy(
        self,
        paths: list[Path],
        mode: Literal['static', 'dynamic', 'fingerprint']='static',
    ) -> list[float]:
        pass

    @action 
    async def cluster_data(
        self,
        data_directory: Path,
        file_name_pattern: str,
        max_clusters: int=10,
        stride: int=1,
    ) -> dict[Path, np.ndarray]:
        pass

    @action
    async def compute_covariance_ppi(
        self,
    ) -> None:
        pass

    @action
    async def run_analyses(
        self,
        manifest: dict[str, dict[str, Any]],
    ) -> None:
        pass
