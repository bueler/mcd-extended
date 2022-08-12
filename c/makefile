include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratu: bratu.o
	-${CLINKER} -o bratu bratu.o ${PETSC_LIB}
	${RM} bratu.o

bratufd: bratufd.o
	-${CLINKER} -o bratufd bratufd.o ${PETSC_LIB}
	${RM} bratufd.o

# testing
runbratufd_1:
	-@./testit.sh bratufd "-lb_exact -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 2 -snes_monitor_short" 1 1

runbratufd_2:
		-@./testit.sh bratufd "-lb_exact -snes_converged_reason -lb_showcounts -snes_view" 2 2

test_bratufd: runbratufd_1 runbratufd_2

runbratu_1:
	-@./testit.sh bratu "-lb_exact -snes_converged_reason -lb_showcounts -snes_monitor_short -ksp_converged_reason -da_refine 1 -pc_type mg" 1 1

test_bratu: runbratu_1

test: test_bratufd test_bratu

# etc

.PHONY: distclean test test_bratufd runbratufd_1

distclean:
	@rm -f bratufd *tmp
