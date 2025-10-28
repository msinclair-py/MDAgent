from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from agents import Builder, MDSimulator, FreeEnergy, MDCoordinator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

async def main():
    init_logging('INFO')

    cwd = Path.cwd() / 'test_mmpbsa2'
    cwd.mkdir(exist_ok=True)

    input_pdb = Path('data/barnase_barnstar.pdb').resolve()
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
        'amberhome': Path(os.environ['AMBERHOME']) / 'bin'
    }

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
            args=(build_agent, sim_agent, free_energy)
        )

        results = await coordinator.deploy_md(
            cwd,
            input_pdb,
            build_kwargs,
            sim_kwargs,
            fe_kwargs,
        )

        print(results)

if __name__ == '__main__':
    asyncio.run(main())
