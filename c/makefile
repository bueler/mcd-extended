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
	-@./testit.sh obstaclesl "-pc_type mg -snes_converged_reason -ksp_converged_reason -snes_grid_sequence 2" 1 1

runobstaclesl_2:
	-@./testit.sh obstaclesl "-ob_pngs -ob_counts -snes_type nrichardson -npc_snes_type ngs -snes_converged_reason" 1 2

test_obstaclesl: runobstaclesl_1 runobstaclesl_2

test: test_bratufd test_bratu test_obstaclesl

.PHONY: distclean test test_bratufd test_bratu test_obstaclesl runbratufd_1 runbratufd_2 runbratufd_3 runbratufd_4 runbratu_1 runbratu_2 runbratu_3 runbratu_4 runobstaclesl_1 runobstaclesl_2

distclean:
	@rm -f bratufd bratu obstaclesl *tmp
