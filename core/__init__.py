"""Shared tools for exploring model activation spaces."""

from core.activations import ActivationSet, ActivationSource
from core.clustering import ClusterAssignments, KMeansClusterer
from core.explorer import ActivationSpaceExplorer, AnalysisResult, PointAnalysis
from core.extraction import ActivationExtractor, ExtractionSpec
from core.plotting import plot_cluster, plot_global
from core.projection import PCAProjector, Projection
from core.runs import ExperimentPlan, RunContext, Stage, run_experiment
from core.version import __version__, repository_state, require_version

__all__ = [
    "ActivationExtractor",
    "ActivationSet",
    "ActivationSource",
    "ActivationSpaceExplorer",
    "AnalysisResult",
    "ClusterAssignments",
    "ExperimentPlan",
    "ExtractionSpec",
    "KMeansClusterer",
    "PCAProjector",
    "PointAnalysis",
    "Projection",
    "RunContext",
    "Stage",
    "__version__",
    "plot_cluster",
    "plot_global",
    "repository_state",
    "require_version",
    "run_experiment",
]
