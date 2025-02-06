def get_task(name='game24'):
    if name == 'game24':
        from tasks.game24 import Game24Task
        return Game24Task()
    else:
        raise NotImplementedError