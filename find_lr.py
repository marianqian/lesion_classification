"""
https://github.com/titu1994/keras-one-cycle
For finding optimal lr while using cyclical learning rate
"""
from clr import LRFinder

# Exponential lr finder
# USE THIS FOR A LARGE RANGE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
lr_finder = LRFinder(train.n, bs, minimum_lr=1e-3, maximum_lr=1.,
                     lr_scale='exp',
                     # validation_data=(X_test, Y_test),  # use the validation data for losses
                     #validation_sample_rate=5,
                     save_dir='onecycle/', verbose=True)



# Linear lr finder
# USE THIS FOR A CLOSE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
# lr_finder = LRFinder(num_samples, batch_size, minimum_lr=5e-4, maximum_lr=1e-2,
#                      lr_scale='linear',
#                      validation_data=(X_test, y_test),  # use the validation data for losses
#                      validation_sample_rate=5,
#                      save_dir='onecycle/', verbose=True)

historylr = model.fit_generator(train, steps_per_epoch= train.n// bs, epochs=1, callbacks = [lr_finder])
    
# plot the previous values if present
LRFinder.plot_schedule_from_file('onecycle/', clip_beginning=10, clip_endding=5)

""""
Learning rate to choose: Find largest lr with lowest loss, before loss function begins to grow and become chaotic
Because the x-axis is in logarithmic form, the maximum learning rate you should use is 10^(value from the graph)
"""
