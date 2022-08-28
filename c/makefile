include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratufd: bratufd.o
	-${CLINKER} -o bratufd bratufd.o ${PETSC_LIB}
	${RM} bratufd.o

bratu: bratu.o q1fem.o
	-${CLINKER} -o bratu bratu.o q1fem.o ${PETSC_LIB}
	${RM} bratu.o q1fem.o

obstaclesl: obstaclesl.o q1fem.o
	-${CLINKER} -o obstaclesl obstaclesl.o q1fem.o ${PETSC_LIB}
	${RM} obstaclesl.o q1fem.o

dcsl: dcsl.o q1fem.o ldc.o
	-${CLINKER} -o dcsl dcsl.o q1fem.o ldc.o ${PETSC_LIB}
	${RM} dcsl.o q1fem.o ldc.o

ldctwolevel: ldctwolevel.o ldc.o
	-${CLINKER} -o ldctwolevel ldctwolevel.o ldc.o ${PETSC_LIB}
	${RM} ldctwolevel.o ldc.o

# testing
runbratufd_1:
	-@./testit.sh bratufd "-lb_exact -snes_converged_reason -lb_counts -ksp_converged_reason -da_refine 1 -pc_type mg" 1 1

runbratufd_2:
	-@./testit.sh bratufd "-lb_exact -snes_converged_reason -lb_counts -snes_type nrichardson -npc_snes_type ngs -npc_snes_max_it 4 -npc_snes_ngs_sweeps 2 -da_refine 1" 1 2

runbratufd_3:
	-@./testit.sh bratufd "-lb_mms -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_max_it 1 -fas_coarse_snes_type newtonls -fas_coarse_snes_fd_color -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_rtol 1.0e-3 -da_refine 2" 1 3

runbratufd_4:
	-@./testit.sh bratufd "-lb_exact -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 2 -snes_monitor_short" 1 4

test_bratufd: runbratufd_1 runbratufd_2 runbratufd_3 runbratufd_4

runbratu_1:
	-@./testit.sh bratu "-lb_exact -snes_converged_reason -lb_counts -ksp_converged_reason -da_refine 1 -pc_type mg" 1 1

runbratu_2:
	-@./testit.sh bratu "-lb_exact -snes_converged_reason -lb_counts -snes_type nrichardson -npc_snes_type ngs -npc_snes_max_it 4 -npc_snes_ngs_sweeps 2 -da_refine 1" 1 2

runbratu_3:
	-@./testit.sh bratu "-lb_mms -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_max_it 1 -fas_coarse_snes_type newtonls -fas_coarse_snes_fd_color -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_rtol 1.0e-3 -da_refine 2" 1 3

runbratu_4:
	-@./testit.sh bratu "-lb_exact -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 2 -snes_monitor_short" 1 4

test_bratu: runbratu_1 runbratu_2 runbratu_3 runbratu_4

runobstaclesl_1:
	-@./testit.sh obstaclesl "-ob_counts -pc_type mg -snes_converged_reason -ksp_converged_reason -snes_grid_sequence 2" 1 1

runobstaclesl_2:
	-@./testit.sh obstaclesl "-ob_counts -ob_pngs -snes_converged_reason -snes_grid_sequence 2" 1 2

runobstaclesl_3:
	-@./testit.sh obstaclesl "-ob_pngs -da_refine 1 -ob_view tmpview.m" 1 3

test_obstaclesl: runobstaclesl_1 runobstaclesl_2 runobstaclesl_3

rundcsl_1:
	-@./testit.sh dcsl "-dc_counts -snes_converged_reason -snes_grid_sequence 2" 1 1

test_dcsl: rundcsl_1

runldctwolevel_1:
	-@./testit.sh ldctwolevel "" 1 1

test_ldctwolevel: runldctwolevel_1

test: test_bratufd test_bratu test_obstaclesl test_dcsl test_ldctwolevel

.PHONY: clean test test_bratufd test_bratu test_obstaclesl test_dcsl test_ldctwolevel runbratufd_1 runbratufd_2 runbratufd_3 runbratufd_4 runbratu_1 runbratu_2 runbratu_3 runbratu_4 runobstaclesl_1 runobstaclesl_2 runobstaclesl_3 rundcsl_1 runldctwolevel_1

clean::
	@rm -f bratufd bratu obstaclesl dcsl ldctwolevel *tmp tmpview.m
