"""
reference
Protein-Ligand Dissociation Simulated by Parallel Cascade Selection Molecular Dynamics
https://doi.org/10.1021/acs.jctc.7b00504
"""

import subprocess
from typing import List

import numpy as np
from pacs.mdrun.analyzer.superAnalyzer import SuperAnalyzer
from pacs.models.settings import MDsettings, Snapshot
from pacs.utils.logger import generate_logger

LOGGER = generate_logger(__name__)


class Dissociation2(SuperAnalyzer):
    def calculate_cv(
        self, settings: MDsettings, cycle: int, replica: int, send_rev
    ) -> List[float]:
        if settings.analyzer == "gromacs":
            ret = self.cal_by_gmx(settings, cycle, replica)
        else:
            raise NotImplementedError
        send_rev.send(ret)
        return ret

    def ranking(self, settings: MDsettings, CVs: List[Snapshot]) -> List[Snapshot]:
        sorted_cv = sorted(CVs, key=lambda x: x.cv, reverse=True)
        return sorted_cv

    def is_threshold(self, settings: MDsettings, CVs: List[Snapshot] = None) -> bool:
        if CVs is None:
            CVs = self.CVs
        return CVs[0].cv >= settings.threshold - settings.threshold2

    def cal_by_gmx(self, settings: MDsettings, cycle: int, replica: int) -> List[float]:
        extension = settings.trajectory_extension
        grp1 = settings.selection1
        grp2 = settings.selection2
        grp3 = settings.selection3
        grp4 = settings.selection4

        dir = settings.each_replica(_cycle=cycle, _replica=replica)

        cmd_image = f"echo 'nowation' \
                | {settings.cmd_gmx} trjconv \
                -f {dir}/prd{extension} \
                -s {dir}/prd.tpr \
                -o {dir}/prd_image{extension} \
                -n {settings.index_file} \
                -pbc mol \
                -ur compact \
                1> {dir}/center.log 2>&1"  # NOQA: E221
        res_image = subprocess.run(cmd_image, shell=True)
        if res_image.returncode != 0:
            LOGGER.error("error occured at image command")
            LOGGER.error(f"see {dir}/center.log")
            exit(1)

        def cal_distance(grp1: str, grp2: str, prefix: str):
            cmd_dist = f"{settings.cmd_gmx} distance \
                    -f {dir}/prd_image{extension} \
                    -s {dir}/prd.tpr \
                    -n {settings.index_file} \
                    -oxyz {dir}/interCOM_xyz_{prefix}.xvg \
                    -xvg none \
                    -select 'com of group {grp1} plus com of group {grp2}' \
                    1> {dir}/distance_{prefix}.log 2>&1"  # NOQA: E221
            res_dist = subprocess.run(cmd_dist, shell=True)
            if res_dist.returncode != 0:
                LOGGER.error("error occurred at distance command")
                LOGGER.error(f"see {dir}/distance.log")
                exit(1)
        cal_distance(grp1, grp2, "1")
        cal_distance(grp3, grp4, "2")

        #cmd_rmfile = f"rm {dir}/prd_image{extension}"
        #subprocess.run(cmd_rmfile, shell=True)

        dist1 = np.linalg.norm(np.loadtxt(f"{dir}/interCOM_xyz_1.xvg")[:, [1, 2, 3]], axis=1)
        dist2 = np.linalg.norm(np.loadtxt(f"{dir}/interCOM_xyz_2.xvg")[:, [1, 2, 3]], axis=1)
        if dist1 > settings.threshold:
            dist1 = settings.threshold
        if dist2 < settings.threshold2:
            dist2 = settings.threshold2

        return dist1 - dist2
