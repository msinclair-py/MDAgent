from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from agents import Builder, MDSimulator, MDCoordinator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

async def main():
    init_logging('INFO')

    cwd = Path.cwd() / 'test_sim'
    cwd.mkdir(exist_ok=True)

    input_pdb = '1UBQ.pdb'
    solvent = 'implicit'

    build_kwargs = {
        'solvent': solvent,
        'protein': True,
        'amberhome': os.environ['AMBERHOME'],
    }
    
    sim_kwargs = {
        'solvent': solvent,
        'equil_steps': 10_000,
        'prod_steps': 1_000_000,
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

        coordinator = await manager.launch(
            MDCoordinator,
            args=(build_agent, sim_agent)
        )

        results = await coordinator.deploy_md(
            input_pdb,
            build_kwargs,
            sim_kwargs,
        )

        print(results)
