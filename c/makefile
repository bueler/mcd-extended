include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratu: bratu.o src/utilities.o src/q1fem.o
	-${CLINKER} -o bratu bratu.o src/utilities.o src/q1fem.o ${PETSC_LIB}
	${RM} bratu.o src/utilities.o src/q1fem.o

obstaclesl: obstaclesl.o src/q1fem.o
	-${CLINKER} -o obstaclesl obstaclesl.o src/q1fem.o ${PETSC_LIB}
	${RM} obstaclesl.o src/q1fem.o

nmcd: nmcd.o src/utilities.o src/q1fem.o src/q1transfers.o src/ldc.o
	-${CLINKER} -o nmcd nmcd.o src/utilities.o src/q1fem.o src/q1transfers.o src/ldc.o ${PETSC_LIB}
	${RM} nmcd.o src/utilities.o src/q1fem.o src/q1transfers.o src/ldc.o

ldctwolevel: ldctwolevel.o src/utilities.o src/q1transfers.o src/ldc.o
	-${CLINKER} -o ldctwolevel ldctwolevel.o src/utilities.o src/q1transfers.o src/ldc.o ${PETSC_LIB}
	${RM} ldctwolevel.o src/utilities.o src/q1transfers.o src/ldc.o

transfers: transfers.o src/utilities.o src/q1transfers.o src/ldc.o
	-${CLINKER} -o transfers transfers.o src/utilities.o src/q1transfers.o src/ldc.o ${PETSC_LIB}
	${RM} transfers.o src/utilities.o src/q1transfers.o src/ldc.o

# testing
runbratu_1: # FD, Newton-Krlov-multigrid
	-@./testit.sh bratu "-lb_fd  -lb_exact -snes_converged_reason -lb_counts -ksp_converged_reason -da_refine 1 -pc_type mg" 1 1

runbratu_2: # FEM, Newton-Krlov-multigrid
	-@./testit.sh bratu "-lb_fem -lb_exact -snes_converged_reason -lb_counts -ksp_converged_reason -da_refine 1 -pc_type mg" 1 2

runbratu_3: # FD, NGS sweeps as solver, parallel
	-@./testit.sh bratu "-lb_fd  -lb_exact -snes_converged_reason -lb_counts -snes_type nrichardson -npc_snes_type ngs -npc_snes_ngs_max_it 4 -npc_snes_ngs_sweeps 2 -da_refine 1" 2 3

runbratu_4: # FEM, NGS sweeps as solver, parallel
	-@./testit.sh bratu "-lb_fem -lb_exact -snes_converged_reason -lb_counts -snes_type nrichardson -npc_snes_type ngs -npc_snes_ngs_max_it 4 -npc_snes_ngs_sweeps 2 -da_refine 1" 2 4

runbratu_5: # FD, FAS-multigrid V-cycles, NGS smoother, Newton-Krylov-CGICC coarse solver
	-@./testit.sh bratu "-lb_fd  -lb_mms -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 1 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type newtonls -fas_coarse_snes_fd_color -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_rtol 1.0e-3 -da_refine 2" 1 5

runbratu_6: # FEM, FAS-multigrid V-cycles, NGS smoother, Newton-Krylov-CGICC coarse solver
	-@./testit.sh bratu "-lb_fem -lb_mms -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 1 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type newtonls -fas_coarse_snes_fd_color -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_rtol 1.0e-3 -da_refine 2" 1 6

runbratu_7: # FD, FAS-multigrid full cycles, NGS smoother, Richardson with NGS as NPC coarse solver
	-@./testit.sh bratu "-lb_fd  -lb_exact -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type nrichardson -fas_coarse_snes_linesearch_type basic -fas_coarse_npc_snes_type ngs -fas_coarse_npc_snes_ngs_sweeps 2 -fas_coarse_npc_snes_ngs_max_it 4 -da_refine 2 -snes_monitor_short" 1 7

runbratu_8: # FEM, FAS-multigrid full cycles, NGS smoother, Richardson with NGS as NPC coarse solver
	-@./testit.sh bratu "-lb_fem -lb_exact -snes_converged_reason -lb_counts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type nrichardson -fas_coarse_snes_linesearch_type basic -fas_coarse_npc_snes_type ngs -fas_coarse_npc_snes_ngs_sweeps 2 -fas_coarse_npc_snes_ngs_max_it 4 -da_refine 2 -snes_monitor_short" 1 8

runbratu_9: # FEM, FAS-multigrid V-cycles, NJacobi smoother, Newton-Krylov-CGICC coarse solver
	-@./testit.sh bratu "-lb_fem -lb_mms -snes_converged_reason -lb_counts -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 1 -fas_levels_snes_ngs_max_it 1 -fas_coarse_snes_type newtonls -fas_coarse_snes_fd_color -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_rtol 1.0e-3 -da_refine 2 -lb_njac" 1 9

test_bratu: runbratu_1 runbratu_2 runbratu_3 runbratu_4 runbratu_5 runbratu_6 runbratu_7 runbratu_8 runbratu_9

runobstaclesl_1:
	-@./testit.sh obstaclesl "-ob_initialzero -ob_counts -pc_type mg -snes_converged_reason -ksp_converged_reason -snes_grid_sequence 2" 1 1

runobstaclesl_2:
	-@./testit.sh obstaclesl "-ob_initialzero -ob_counts -ob_pngs -snes_converged_reason -snes_grid_sequence 2" 1 2

runobstaclesl_3:
	-@./testit.sh obstaclesl "-ob_initialzero -ob_pngs -da_refine 1 -ob_view tmpview.m" 1 3

# compare following run to single-level run of nmcd.c in runnmcd_3
runobstaclesl_4:
	-@./testit.sh obstaclesl "-ob_initialbump 1.0 -da_refine 1 -ob_pngs -snes_converged_reason -npc_snes_ngs_max_it 1 -snes_monitor" 1 4

test_obstaclesl: runobstaclesl_1 runobstaclesl_2 runobstaclesl_3 runobstaclesl_4

runnmcd_1:
	-@./testit.sh nmcd "-nm_monitor_ranges -nm_counts -nm_ldcinfo -nm_levels 3 -da_grid_x 4 -da_grid_y 4" 1 1

runnmcd_2:
	-@./testit.sh nmcd "-nm_monitor_ranges -nm_monitor_vcycles -nm_counts -nm_bumpsize 1.0 -nm_levels 4 -nm_cycles 2" 1 2

# compare following run to single-level run of obstaclesl.c in runobstaclesl_4
runnmcd_3:
	-@./testit.sh nmcd "-nm_monitor -nm_bumpsize 1.0 -da_refine 1 -nm_levels 1 -nm_cycles 4" 1 3

test_nmcd: runnmcd_1 runnmcd_2 runnmcd_3

runldctwolevel_1:
	-@./testit.sh ldctwolevel "-unilateral" 1 1

# FIXME add bilateral ldctwolevel regression, after checking both unilateral and bilateral stdout

test_ldctwolevel: runldctwolevel_1

runtransfers_1:
	-@./testit.sh transfers "" 1 1

test_transfers: runtransfers_1

test: test_bratu test_obstaclesl test_nmcd test_ldctwolevel test_transfers

.PHONY: clean test test_bratu test_obstaclesl test_nmcd test_ldctwolevel test_transfers runbratu_1 runbratu_2 runbratu_3 runbratu_4 runbratu_5 runbratu_6 runbratu_7 runbratu_8 runbratu_9 runobstaclesl_1 runobstaclesl_2 runobstaclesl_3 runobstaclesl_4 runnmcd_1 runnmcd_2 runnmcd_3 runldctwolevel_1 runtransfers_1

clean::
	@rm -f bratu obstaclesl nmcd ldctwolevel transfers *tmp tmpview.m
