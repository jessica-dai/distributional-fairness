import numpy as np
import itertools
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

def group_loss(x, group, g, t, ep):
    num_groups = g.shape[0]

    # input is a vector of lambdas
    terms = []

    fA = g[group]
    lA = x[group]
    tA = t[group]

    for i in range(num_groups):
        if i != group:
            fB = g[i]
            lB = x[i]
            tB = t[i]

            terms.append(np.abs(((fA + lA * tA) - (fB + lB * tB))))

    return np.sum(np.array(terms)) + ep

#Computes the total loss across groups with respect to some specific group(unweighted)
def group_loss_(x, groups, g, t, fudge, ep):
    return np.sum(np.array([group_loss(x, group, g, t, ep) for group in groups])) + fudge


def lexicographicOptimizer(df, attr_col, score_col, shift_col, ep=1e-6):

    df = df.copy().sort_values(by=attr_col) # added jess 1.14.22 - make group order deterministic

    group_shift = [d[1][shift_col].values.copy() for d in
                   df.groupby(attr_col)]  # collect all the scores for each group from the data
    avg_shift = np.array([shift.mean() for shift in group_shift])  # get average shift
    group_scores = [d[1][score_col].values.copy() for d in
                    df.groupby(attr_col)]  # collect all the scores for each group from the data
    avg_score = np.array([scores.mean() for scores in group_scores])  # compute the peo score for each group by averaging scores

    alpha = 1e-4
    num_groups = avg_shift.shape[0]

    I = np.identity(num_groups)

    #Compute group losses with no repair, ie barycenter pi=0 for all i
    #these are all iterates, this is their "initial" value
    group_losses = np.array([group_loss(np.zeros(num_groups), i, g=avg_score, t=avg_shift, ep=ep) for i in np.arange(num_groups)])
    # compute group losses with full repair
    full_losses = np.array([group_loss(np.ones(num_groups), i, g=avg_score, t=avg_shift, ep=ep) for i in np.arange(num_groups)])
    # Outermost loop for the number of groups that we have
    eta = []
    lambdas_rec = []
    groupwise_losses = []

    groupwise_losses.append(group_losses)
    groupwise_losses.append(full_losses)

    for j in range(num_groups):
        # Step 1. set up constraints in this for loop

        #this will restrict lambda to be between 0, 1
        cons = [LinearConstraint(I[ix], 0, 1) for ix in range(num_groups)]  # basic bounds
        if j > 0:
            for r in range(j):
                # Get all permutations of the desired length, need this for all the subset stuff
                perms = list(itertools.permutations(np.arange(0, num_groups), r + 1))
                # #get the lexicographic constraints

                for perm in perms:
                    if len(perm) == 1:
                        for p in perm:
                            constraint = NonlinearConstraint(lambda x: group_loss(x, p, g=avg_score, t=avg_shift, ep=0), lb=0,
                                                             ub=eta[r])
                            cons.append(constraint)
                    else:
                        constraint = NonlinearConstraint(
                            lambda x: np.array(group_loss_(x, perm, g=avg_score, t=avg_shift, fudge=alpha, ep=ep)), lb=0, ub=eta[r])
                        cons.append(constraint)

        # Step 2. Solve the optimization

        # Find worst group(s) with present adjustment
        worst_off = np.argpartition(group_losses, num_groups - (j + 1))[-(j + 1):]  # get indices of top j losses

        res = minimize(
            fun=group_loss_,
            x0=np.ones(num_groups) / num_groups,
            args=(worst_off, avg_score, avg_shift, alpha, ep),
            method='trust-constr',
            constraints=cons,
            options={"disp": True}
        )

        # Update group losses
        group_losses = np.array([group_loss(res.x, i, g=avg_score, t=avg_shift, ep=ep) for i in np.arange(num_groups)])

        # Collect worst case eta
        perms = list(itertools.permutations(np.arange(0, num_groups), j + 1))

        if j == 0:
            eta_losses = [group_loss(res.x, p[0], avg_score, avg_shift, ep) for p in perms]
        else:
            eta_losses = [group_loss_(res.x, p, avg_score, avg_shift, alpha, ep) for p in perms]
        eta.append(min(eta_losses))


        # group_losses_ = np.array([group_loss(res.x, j, g=avg_score, t=avg_shift, ep=ep) for j in range(num_groups)])
        groupwise_losses.append(group_losses)
        lambdas = res.x
        lambdas_rec.append(lambdas)

    group_map = {list(df.groupby(attr_col).groups.keys())[i] : i for i in np.arange(num_groups)}

    return np.array(lambdas_rec), group_map, np.array(groupwise_losses)

    df["lexi  "] = df.apply(lambda x: x[score_col] + x[shift_col]*lambdas_rec[-1][group_map[x[attr_col]]], axis=1)
    df["maxmin"] = df.apply(lambda x: x[score_col] + x[shift_col]*lambdas_rec[-1][group_map[x[attr_col]]], axis=1)

    # return df.copy()

