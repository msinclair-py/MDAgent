from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from MDAgent.core.agents_parsl_fixed import Builder, MDSimulator, FreeEnergy, MDCoordinator
import os
import parsl
from parsl import Config, HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiExecLauncher
from pathlib import Path

async def main():
    init_logging('INFO')

    cwd = Path.cwd()

    paths = []
    for i in range(4):
        path = cwd / f'trial_{i}'
        path.mkdir(exist_ok=True)
        paths.append(path)

    input_pdbs = [Path('data/barnase_barnstar.pdb').resolve() for _ in range(4)]

    solvent = 'explicit'

    build_kwargs = {
        'solvent': solvent,
        'protein': True,
        'out': 'system.pdb'
    }
    
    sim_kwargs = {
        'solvent': solvent,
        'equil_steps': 10_000,
        'prod_steps': 100_000,
    }

    fe_kwargs = {
        'selections': [':1-110', ':111-200'],
        'n_cpus': 64,
        'amberhome': Path(os.environ['AMBERHOME'])
    }

    build_kwargss = [build_kwargs for _ in range(4)]
    sim_kwargss = [sim_kwargs for _ in range(4)]
    fe_kwargss = [fe_kwargs.copy() for _ in range(4)]

    parsl_config = Config(
        executors=[
            HighThroughputExecutor(
                label='agentic_MD',
                cores_per_worker=4,
                worker_debug=True,
                provider=LocalProvider(parallelism=1, max_blocks=4),
                max_workers_per_node=4,
                available_accelerators=['0', '1', '2', '3']
            ),
        ],
    )

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        build_agent = await manager.launch(
            Builder,
        )

        sim_agent = await manager.launch(
            MDSimulator,
        )

        free_energy = await manager.launch(
            FreeEnergy,
        )

        coordinator = await manager.launch(
            MDCoordinator,
            args=(build_agent, sim_agent, free_energy, parsl_config)
        )

        results = await coordinator.deploy_md(
            paths,
            input_pdbs,
            build_kwargss,
            sim_kwargss,
            fe_kwargss,
        )

        print(results)

if __name__ == '__main__':
    asyncio.run(main())
