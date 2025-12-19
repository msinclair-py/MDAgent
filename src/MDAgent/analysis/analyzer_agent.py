from academy.agent import Agent, action
from academy.handle import Handle
import asyncio
from collections import ChainMap
import logging
import parsl
from parsl import python_app, Config
from pathlib import Path
from typing import Any 

logger = logging.getLogger(__name__)

# Parsl apps that directly execute the simulation tasks
# These run in separate processes/threads and don't use the Agent framework
@python_app
def parsl_static(_type: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute build task directly in Parsl worker."""
    from .analysis_workflows import basic_static_workflow, advanced_static_workflow

    match _type:
        case 'basic':
            result = basic_static_workflow(arguments['paths'])
        case 'advanced':
            result = advanced_static_workflow(arguments['paths'], arguments['kwargs'])
        case _:
            logger.warning(f'Analysis type {_type} not implemented!')
            result = None
    
    return result

@python_app
def parsl_dynamic(analysis: dict[str, Any]) -> dict[str, Any]:
    """Execute simulation task directly in Parsl worker."""
    from .analysis_workflows import basic_simulation_workflow, advanced_simulation_workflow
    
    match _type:
        case 'basic':
            result = basic_simulation_workflow(arguments['paths'])
        case 'advanced':
            result = advanced_simulation_workflow(arguments['paths'], arguments['kwargs'])
        case _:
            logger.warning(f'Analysis type {_type} not implemented!')
            result = None
    
    return result

class AnalysisCoordinator(Agent):
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
    async def analyze(
        self,
        analysis_plan: dict[str, Any]:
    ) -> list[dict[str, Any]]:
        """Deploy full analysis workflow using Parsl.
        Input structure should be as follows:
            {
            'static': {
                'type0': {
                        'paths': [path0, path1, ...],
                        'kwargs': {'var': value, ...},
                        }

            },
            'dynamic': {
                'type0': {
                        'paths': [path0, path1, path2, ...],
                        'kwargs': {'var': value, ...},
                        },
                'type1': {
                        'paths': [path0, path1, ...],
                        'kwargs': {'var': value, ...},
                        },
            },
            }
        """
        static_analyses = analysis_plan.get('static', {})
        dynamic_analyses = analysis_plan.get('dynamic', {})

        static = []
        if static_analyses:
            futures = []
            for _type, arguments in static_analyses.items():
                futures.append(asyncio.wrap_future(parsl_static(_type, arguments)))

            static += await asyncio.gather(*futures)
            static = dict(ChainMap(*static))
        else:
            static = None
            
        
        dynamic = []
        if dynamic_analyses:
            futures = []
            for _type, arguments in dynamic_analyses.items():
                futures.append(asyncio.wrap_future(parsl_static(_type, arguments)))

            dynamic += await asyncio.gather(*futures)
            dynamic = dict(ChainMap(*dict))
        else:
            dynamic = None
        
        return {'static': static, 'dynamic': dynamic}
