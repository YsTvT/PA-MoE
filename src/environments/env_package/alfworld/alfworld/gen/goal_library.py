

gdict = {}

gdict["pick_and_place_simple"] = \
{
    'pddl' :
    ,
    'templates': ['put a {obj} in {recep}',
                   'put some {obj} on {recep}']
}

gdict["pick_clean_then_place_in_recep"] = \
{
    'pddl' :
    ,
    'templates': ['put a clean {obj} in {recep}',
                   'clean some {obj} and put it in {recep}']

}

gdict["pick_heat_then_place_in_recep"] = \
{
    'pddl':
    ,
    'templates': ['put a hot {obj} in {recep}',
                   'heat some {obj} and put it in {recep}']
}

gdict["pick_cool_then_place_in_recep"] = \
{
    'pddl':
    ,
    'templates': ['put a cool {obj} in {recep}',
                   'cool some {obj} and put it in {recep}']
}

gdict["pick_two_obj_and_place"] = \
    {
        'pddl':
            ,
        'templates': ['put two {obj} in {recep}',
                      'find two {obj} and put them in {recep}']
    }

gdict["look_at_obj_in_light"] = \
{
    'pddl':
    ,
    'templates': ['look at {obj} under the {toggle}',
                  'examine the {obj} with the {toggle}']
}

gdict["pick_and_place_with_movable_recep"] = \
    {
        'pddl':
            ,
        'templates': ['put {obj} in a {mrecep} and then put them in {recep}',
                      'put a {mrecep} of {obj} in {recep}',
                      'put {obj} {mrecep} in {recep}']
    }

gdict["pick_clean_then_place_in_recep_slice"] = \
    {
        'pddl':
            ,
        'templates': ['put a clean slice of {obj} in {recep}',
                      'clean some sliced {obj} and put it in {recep}']

    }

gdict["pick_heat_then_place_in_recep_slice"] = \
    {
        'pddl':
            ,
        'templates': ['put a hot slice of {obj} in {recep}',
                      'heat some sliced {obj} and put it in {recep}']
    }

gdict["pick_cool_then_place_in_recep_slice"] = \
    {
        'pddl':
            ,
        'templates': ['put a cool slice of {obj} in {recep}',
                      'cool some sliced {obj} and put it in {recep}']
    }

gdict["pick_two_obj_and_place_slice"] = \
    {
        'pddl':
            ,
        'templates': ['put two sliced {obj} in {recep}',
                      'find two sliced {obj} and put them in {recep}']
    }

gdict["look_at_obj_in_light_slice"] = \
    {
        'pddl':
            ,
        'templates': ['look at sliced {obj} under the {toggle}',
                      'examine the sliced {obj} with the {toggle}']
    }

gdict["pick_and_place_with_movable_recep_slice"] = \
    {
        'pddl':
            ,
        'templates': ['put sliced {obj} in a {mrecep} and then put them in {recep}',
                      'put a {mrecep} of sliced {obj} in {recep}',
                      'put sliced {obj} {mrecep} in {recep}']
    }

gdict["pick_and_place_simple_slice"] = \
    {
        'pddl':
            ,
        'templates': ['slice {obj} and put in {recep}',
                      'put sliced {obj} in {recep}']
    }

gdict["place_all_obj_type_into_recep"] = \
{
    'pddl':
    ,
    'templates': ['put all {obj}s in {recep}',
                   'find all {obj}s and put them in {recep}']
}

gdict["pick_three_obj_and_place"] = \
    {
        'pddl':
            ,
        'templates': ['put three {obj} in {recep}',
                      'find three {obj} and put them in {recep}']
    }

gdict["pick_heat_and_place_with_movable_recep"] = \
    {
        'pddl':
            ,
        'templates': ['put a hot {mrecep} of {obj} in {recep}']
    }

gdict["pick_cool_and_place_with_movable_recep"] = \
    {
        'pddl':
            ,
        'templates': ['put a cold {mrecep} of {obj} in {recep}']
    }

gdict["pick_clean_and_place_with_movable_recep"] = \
    {
        'pddl':
            ,
        'templates': ['put a cold {mrecep} of {obj} in {recep}']
    }