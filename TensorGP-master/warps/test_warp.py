import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
import math

def resolve_var_node(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = tf.reshape(tf.range(dimensions[n], dtype=tf.float32), temp)
    resolution = dimensions[n]
    dimensions[n] = 1
    res = tf.scalar_mul(((1.0 / (resolution - 1)) * 2.0), res)
    res = tf.math.subtract(res, tf.constant(1.0, tf.float32, res.shape))
    res = tf.tile(res, dimensions)
    return res


def resolve_sin_node(child1):
    return tf.math.sin(tf.scalar_mul(math.pi, child1))

def resolve_max_node(child1, child2):
    return tf.math.maximum(child1, child2)

def resolve_div_node(child1, child2):
    left_child_tensor = tf.cast(child1, tf.float32)
    right_child_tensor = tf.cast(child2, tf.float32)
    return tf.math.divide_no_nan(left_child_tensor, right_child_tensor)

def resolve_abs_node(child1):
    return tf.math.abs(child1)

def resolve_add_node(child1, child2):
    return tf.math.add(child1, child2)

def resolve_mult_node(child1, child2):
    return tf.math.multiply(child1, child2)

def resolve_min_node(child1, child2):
    return tf.math.minimum(child1, child2)

def scalar(r, g, b):
    cols = [r, g, b]
    return tf.stack(
        [tf.constant(c, tf.float32, dims[:-1]) for c in cols],
        axis = 2
    )

def normal_warp(image, xcoord_im, ycoord_im, dims, mind, maxd):
    result = np.empty(dims, dtype=float)
    indices = np.empty(dims + [3], dtype=float)

    wid = dims[0]
    hei = dims[1]
    for i in range(wid):
        for j in range(hei):

            xCoord = xcoord_im[i][j]
            yCoord = ycoord_im[i][j]


            auxX = (wid) / (maxd - mind)
            auxY = (hei) / (maxd - mind)

            i1 = int(round((xCoord[0] - mind) * auxX))
            j1 = int(round((yCoord[0] - mind) * auxY))
            i2 = int(round((xCoord[1] - mind) * auxX))
            j2 = int(round((yCoord[1] - mind) * auxY))
            i3 = int(round((xCoord[2] - mind) * auxX))
            j3 = int(round((yCoord[2] - mind) * auxY))

            i1 = (i1 if (i1 >= 0) else 0) if (i1 < wid) else (wid - 1)
            j1 = (j1 if (j1 >= 0) else 0) if (j1 < hei) else (hei - 1)
            i2 = (i2 if (i2 >= 0) else 0) if (i2 < wid) else (wid - 1)
            j2 = (j2 if (j2 >= 0) else 0) if (j2 < hei) else (hei - 1)
            i3 = (i3 if (i3 >= 0) else 0) if (i3 < wid) else (wid - 1)
            j3 = (j3 if (j3 >= 0) else 0) if (j3 < hei) else (hei - 1)

            result[i][j][0] = image[i1][j1][0]
            result[i][j][1] = image[i2][j2][1]
            result[i][j][2] = image[i3][j3][2]

            if i == 0 and j == 0:
                print("(i1, i2, i3) : " + str([i1, i2, i3]))
                print("(i1, i2, i3) : " + str([j1, j2, j3]))


            # for debugging
            indices[i][j][0][0] = i1
            indices[i][j][0][1] = j1
            indices[i][j][0][2] = 0
            indices[i][j][1][0] = i2
            indices[i][j][1][1] = j2
            indices[i][j][1][2] = 1
            indices[i][j][2][0] = i3
            indices[i][j][2][1] = j3
            indices[i][j][2][2] = 2

    print("Indices (normal): ")
    print(indices)

    return result

def resolve_warp_node(tensors, image, dimensions):
    n = len(dimensions)

    indices = tf.stack([
        tf.clip_by_value(
            tf.round(tf.multiply(
                tf.constant((dimensions[k] - 1) * 0.5, tf.float32, shape=dimensions),
                tf.math.add(tensors[k], tf.constant(1.0, tf.float32, shape=dimensions))
            )),
            clip_value_min=0.0,
            clip_value_max=(dimensions[k] - 1)
        ) for k in range(n)],
        axis = n
    )

    indices = tf.cast(indices, tf.int32)
    print("Indices stack (after): ")
    print(indices.numpy())

    return tf.gather_nd(image, indices)

def final_transform(final_tensor, dims):
    final_tensor = tf.clip_by_value(final_tensor, clip_value_min=-1, clip_value_max=1)
    final_tensor =  tf.math.add(final_tensor, tf.constant(1.0, tf.float32, dims))
    return tf.scalar_mul(127.5, final_tensor)



if __name__ == "__main__":
    #expr = 'warp(abs(x), x, mult(add(scalar(0.46855265, 0.46855265, 0.46855265), y), min(scalar(0.9893352, 0.70297724, 0.44974214), x)))'

    save_image = True

    tf.random.set_seed(123456789)
    dims = [256, 256, 3]
    tx = resolve_var_node(np.copy(dims), 1)
    ty = resolve_var_node(np.copy(dims), 0)
    tz = resolve_var_node(np.copy(dims), 2)
    #tr1 = tf.random.uniform(shape=dims, minval = -1, maxval=1, dtype=tf.float32)
    #tr2 = tf.random.uniform(shape=dims, minval=-1, maxval=1, dtype=tf.float32)

    timage1 = resolve_abs_node(tx)
    tx1 = tx
    ty1 = resolve_mult_node(resolve_add_node(scalar(0.46855265, 0.46855265, 0.46855265), ty), resolve_min_node(scalar(0.9893352, 0.70297724, 0.44974214), tx))

    # image 02 with manual warp
    warp = resolve_sin_node(resolve_div_node(tf.convert_to_tensor(normal_warp(ty.numpy(), ty.numpy(), tx.numpy(), dims, -1.0, 1.0), dtype=tf.float32), resolve_max_node(scalar(0.22829413, 0.63101137, 0.4622789), scalar(0.85845333, 0.85845333, 0.85845333))))

    # image 02 with TF warp
    #warp = resolve_sin_node(resolve_div_node(resolve_warp_node([ty, tx, tz], ty, dims), resolve_max_node(scalar(0.22829413, 0.63101137, 0.4622789), scalar(0.85845333, 0.85845333, 0.85845333))))

    warp = final_transform(warp, dims)

    print("Result: ")
    to_arr = warp.numpy()
    print(to_arr)

    if save_image:
        image_name =  "temp_warp.jpg"
        aux = np.array(warp, dtype='uint8')
        Image.fromarray(aux, mode = "RGB").save(image_name)

        # comment this block to do RGB
        im = Image.open(image_name)
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        im.save(image_name)
        im = Image.open(image_name)
        im.show()