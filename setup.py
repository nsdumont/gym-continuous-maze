from setuptools import find_packages, setup

setup(
    name="gym_continuous_maze",
    description="Continuous maze environment integrated with OpenAI/Gym",
    author="Quentin Gallouédec",
    url="https://github.com/qgallouedec/gym-continuous-maze",
    author_email="gallouedec.quentin@gmail.com",
    license="MIT",
    version="0.0.0",
    packages=find_packages(),
    package_data = {'': ["maze_samples/*.npy"]},
    install_requires = ["gymnasium", "pygame", "numpy"]
)
