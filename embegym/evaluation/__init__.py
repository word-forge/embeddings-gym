from . import base, analogies, similarities


def make_full_evaluation_en(n_jobs=1):
    return base.Evaluation((('SimLex999', similarities.SimLex999()),
                            ),
                           n_jobs=n_jobs)
