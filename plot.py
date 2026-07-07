import numpy as np

# RGB recipes:
natural_colour_recipe = {
    'r':dict(
        channel = 'NIR1.6',
        range = (0., 1.),
        gamma = 1.,
    ),
    'g':dict(
        channel = 'VNIR0.8',
        range = (0., 1.),
        gamma = 1.,
    ),
    'b':dict(
        channel = 'VIS0.6',
        range = (0., 1.),
        gamma = 1.,
    )
}


natural_colour_enhanced_recipe = {
    'r':dict(
        channel = 'NIR1.6',
        range = (0., 1.),
        gamma = 3.,
    ),
    'g':dict(
        channel = 'VNIR0.8',
        range = (0., 1.),
        gamma = 3.,
    ),
    'b':dict(
        channel = 'VIS0.6',
        range = (0., 1.),
        gamma = 3.,
    )
}


cloud_phase_daytime_recipe = {
    'r':dict(
        channel = 'NIR1.6',
        range = (0., 1.),
        gamma = 1.,
    ),
    'g':dict(
        channel = 'NIR2.25',
        range = (0., 1.),
        gamma = 1.,
    ),
    'b':dict(
        channel = 'VIS0.6',
        range = (0., 1.),
        gamma = 1.,
    )
}


cloud_microphys_24h_recipe = {
    'r':dict(
        channels = ['IR12.0', 'IR10.8'],
        weights = [1, -1],
        range = (-4., 2.),
        gamma = 1.,
    ),
    'g':dict(
        channels = ['IR10.8', 'IR8.7'],
        weights = [1, -1],
        range = (0., 6.),
        gamma = 1.2,
    ),
    'b':dict(
        channel = 'IR10.8',
        range = (248., 303.),
        gamma = 1.,
    )
}


def _get_channel_image(image_ds, channels, translation):

    images_output = [image_ds.sel(
        { translation['channel_name_dim'] : translation[channel] }).values
        for channel in channels]

    return images_output


def _gamma_scaling(values, range_min, range_max, gamma):

    values_clipped = np.clip(values, range_min, range_max)

    output = np.divide( values_clipped - range_min , range_max - range_min ) ** (1/gamma)

    return output


def rgb_false_colour(image_data, rgb_recipe, instrument_translation):

    image_shape = image_data.shape[1:]

    rgb_output = np.zeros(shape=(*image_shape, 3))

    for idx, colour in enumerate(['r', 'g', 'b']):
        colour_recipe = rgb_recipe[colour]

        if all([key in colour_recipe.keys() for key in ['channels', 'weights']]):
            computed_values = np.sum([pixel_values * weight for pixel_values, weight in 
                                zip(_get_channel_image(image_data, colour_recipe['channels'], instrument_translation),
                                    colour_recipe['weights'])], axis=0)
        
        elif 'channel' in colour_recipe.keys():
            computed_values = np.asarray( _get_channel_image(image_data, [colour_recipe['channel']], instrument_translation) )

        else:
            raise ValueError('Incorrect specification of channels for RGB image recipe')

        colour_values = _gamma_scaling(computed_values, *colour_recipe['range'], colour_recipe['gamma'])

        rgb_output[:,:,idx] = colour_values

    return rgb_output

