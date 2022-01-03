def group_loss(x, group, g, t, ep):
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


def group_loss_(x, groups, g, t, fudge, ep):
    return np.sum(np.array([group_loss(x, group, g, t, ep) for group in groups])) + fudge


def lexicographicOptimizer(test_, ep=1e-6):
    group_shift = [d[1]["shift"].values.copy() for d in
                   test_.groupby("group")]  # collect all the scores for each group from the data
    t = np.array([shift.mean() for shift in group_shift])  # get average shift
    group_scores = [d[1].score.values.copy() for d in
                    test_.groupby("group")]  # collect all the scores for each group from the data
    g = np.array([scores.mean() for scores in group_scores])  # compute the peo score for each group by averaging scores

    alpha = 1e-4
    I = np.identity(num_groups)
    group_losses = np.array([group_loss(np.zeros(num_groups), i, g=g, t=t, ep=ep) for i in np.arange(num_groups)])
    # Outermost loop for the number of groups that we have
    eta = []
    ordering = []
    cons = []
    lambdas_rec = []

    for j in range(3):
        # Step 1. set up constraints in this for loop
        cons = [LinearConstraint(I[ix], 0, 1) for ix in range(num_groups)]  # basic bounds
        if j > 0:
            for r in range(j):
                # Get all permutations of the desired length, need this for all the subset stuff
                perms = list(itertools.permutations(np.arange(0, num_groups), r + 1))
                # #get the lexicographic constraints

                for perm in perms:
                    if len(perm) == 0:
                        continue
                        for p in perm:
                            constraint = NonlinearConstraint(lambda x: group_loss(x, p, g=g, t=t, ep=0), lb=0,
                                                             ub=eta[r])
                            cons.append(constraint)
                    else:
                        constraint = NonlinearConstraint(
                            lambda x: np.array(group_loss_(x, perm, g=g, t=t, fudge=alpha, ep=ep)), lb=0, ub=eta[r])
                        cons.append(constraint)

        # Step 2. Solve the optimization

        # Find worst group(s) with present adjustment
        worst_off = np.argpartition(group_losses, num_groups - (j + 1))[-(j + 1):]  # get indices of top j losses

        res = minimize(
            fun=group_loss_,
            x0=np.ones(num_groups) / num_groups,
            args=(worst_off, g, t, alpha, ep),
            method='trust-constr',
            constraints=cons,
            tol=10e-7,
            options={"disp": True}
        )

        # Update group losses
        # group_losses = np.array([group_loss(res.x, i, g=g, t=t) for i in np.arange(num_groups)])

        # Collect worst case eta
        perms = list(itertools.permutations(np.arange(0, num_groups), j + 1))

        if j == 0:
            eta_losses = [group_loss(res.x, p[0], g, t, ep) for p in perms]
        else:
            eta_losses = [group_loss_(res.x, p, g, t, alpha, ep) for p in perms]
        eta.append(min(eta_losses))
        print("eta losses:", eta_losses)
        print("min eta:", min(eta_losses))

        group_losses_ = np.array([group_loss(res.x, j, g=g, t=t, ep=ep) for j in range(num_groups)])
        lambdas = res.x
        lambdas_rec.append(lambdas)

    lexi_group_losses_ = np.array([group_loss(lambdas_rec[-1], j, g=g, t=t, ep=0) for j in range(num_groups)])
    mm_group_losses_ = np.array([group_loss(lambdas_rec[0], j, g=g, t=t, ep=0) for j in range(num_groups)])
    sdp_group_losses_ = np.array([group_loss(np.ones(num_groups), j, g=g, t=t, ep=0) for j in range(num_groups)])
    unconst_group_losses_ = np.array([group_loss(np.zeros(num_groups), j, g=g, t=t, ep=0) for j in range(num_groups)])
    return {
        "lexi": lexi_group_losses_,
        "maxmin": mm_group_losses_,
        "sdp": sdp_group_losses_,
        "unconst": unconst_group_losses_,
        "lambdas": lambdas_rec
    }


def map_to_barycenter(bc_cdf, input_data):
    groups = [d[1] for d in input_data.groupby('group')]
    group_percentiles = [vect_to_cdf(g.score) for g in groups]

    new_scores = []
    for ix, row in input_data.iterrows():
        S = int(row.group)
        score = row['score']
        score_percentile = group_percentiles[S][np.where(scores == score)[0][0]]
        new_score_ix = np.argmin(np.abs(bc_cdf - score_percentile))
        new_scores.append(scores[new_score_ix])

    new_scores = np.array(new_scores)

    input_data["adjusted"] = new_scores
    input_data["shift"] = np.array(input_data["adjusted"] - input_data["score"])
    return input_data


def computeBarycenter(A, weights):
    A = FICO['pdf'].to_numpy()
    dim, num_groups = FICO['pdf'].to_numpy().shape

    M = ot.utils.dist0(dim)
    M /= M.max()

    weights = np.array(list(FICO['proportions'].values()))
    # wasserstein
    alpha = 0.2
    reg = 1e-3
    bc = ot.bregman.barycenter(A, M, reg, weights)
    return bc


def vect_to_cdf(vect):
    return [(vect <= score).astype(int).mean() for score in scores]


def get_pdfs(data):
    groups = [d[1] for d in data.groupby('group')]
    hist = [np.histogram(g.score.values, np.arange(0, 100.5, 0.5))[0] / len(g) for g in groups]
    return np.array(hist)