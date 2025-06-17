project('NN', 'cpp', default_options : ['cpp_std=c++20'])

gtest_subproj = subproject('gtest')
gtest_dep = gtest_subproj.get_variable('gtest_main_dep')

omp_dep = dependency('openmp', required: true)

NN_dir = include_directories('../../include/')

e = executable(
    'NNtest',
    'tests.cpp',
    include_directories : NN_dir,
    dependencies : [gtest_dep, omp_dep]
)
test('FFTtest', e)
