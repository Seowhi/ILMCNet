
import sys


class ProgressBar(object):



    def __init__(self, epoch_size, batch_size, max_arrow=80):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.max_steps = round(epoch_size / batch_size)
        self.max_arrow = max_arrow


    def show_process(self, train_acc, train_loss, f1, used_time, i):
        num_arrow = int(i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = i * 100.0 / self.max_steps
        num_steps = self.batch_size * i
        process_bar =  '%d'%num_steps + '/' + '%d'%self.epoch_size + '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + ' - train_acc ' + '%.4f'%train_acc + ' - train_loss '+ \
                       '%.4f' %train_loss + ' - f1 ' + '%.4f'% f1 + ' - time '+ '%.1fs'%used_time + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
