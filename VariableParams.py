from skimage.morphology import disk


promptdict1 = {
    1: "blue sky with clouds",
    2: "rocks and sand",
    3: "vegetation and trees",
    4: "water with reflection of sky",
    5: "human being",
    6: "inanimate object",
    7: "building",
    8: "stained glass window",
    9: "stained glass window"
}


promptdict2 = {
    1: "cloudy sky",
    2: "ground",
    3: "vegetation and trees",
    4: "water with reflection of sky",
    5: "human being",
    6: "inanimate object",
    7: "road",
    8: "blinds on a window",
    9: "blinds on a window"
}


promptdict3 = {
    1: "blue sky",
    2: "ground",
    3: "vegetation and trees",
    4: "water with reflection of sky",
    5: "human being",
    6: "inanimate object",
    7: "pavement",
    8: "table lamp with glowing bulb",
    9: "table lamp with glowing bulb"
}


promptdict4 = {
    1: "blue sky with clouds",
    2: "rocks and sand",
    3: "vegetation and trees",
    4: "water with reflection of sky",
    5: "human being",
    6: "inanimate object",
    7: "pavement with cobbled stones",
    8: "view of blue sky with clouds",
    9: "view of blue sky with clouds"
}


promptdict5 = {
    1: "clear blue sky",
    2: "rocks and sand",
    3: "vegetation and trees",
    4: "fill with blue water",
    5: "human being",
    6: "inanimate object",
    7: "pavement with cobbled stones",
    8: "view of building across the street",
    9: "view of building across the street"
}
promptdict6 = {
    1: "clear sky",
    2: "rocks and sand",
    3: "vegetation and trees",
    4: "blue water",
    5: "human being",
    6: "inanimate object",
    7: "pavement with cobbled stones",
    8: "view of building across the street",
    9: "view of building across the street"
}
dicts = [promptdict4, promptdict5]

rad1 = disk(1)
rad2 = disk(2)
rad5 = disk(5)
rad10 = disk(10)

pair1 = (rad5, rad2)
pair2 = (rad10, rad5)
pair3 = (rad5, rad1)
pair4 = (rad2, rad5)

morphology_radius_pairs = [pair1]

legend_dict = ['pd4', 'pd5']
legend_rad = ['rd1']