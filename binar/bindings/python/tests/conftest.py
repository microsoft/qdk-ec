import os
from hypothesis import settings, Verbosity

settings.register_profile("factory")
settings.register_profile("build", print_blob=True, deadline=500)
settings.register_profile("fast", max_examples=10)
settings.register_profile("thorough", max_examples=1000)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))
