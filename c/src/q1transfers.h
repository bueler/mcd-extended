// Tools for Q1 interpolation, restriction, injection, and monotone restriction.

#ifndef Q1TRANSFERS_H_
#define Q1TRANSFERS_H_

typedef enum {
    Q1_MAX, Q1_MIN
} Q1MonotoneType;

PetscErrorCode Q1Interpolate(DM dac, DM daf, Vec vcoarse, Vec *vfine);

PetscErrorCode Q1Restrict(DM daf, DM dac, Vec vfine, Vec *vcoarse);

PetscErrorCode Q1Inject(DM daf, DM dac, Vec vfine, Vec *vcoarse);

PetscErrorCode Q1MonotoneRestrict(Q1MonotoneType opt, DM daf, DM dac,
                                  Vec vfine, Vec *vcoarse);

#endif  // Q1TRANSFERS_H_
