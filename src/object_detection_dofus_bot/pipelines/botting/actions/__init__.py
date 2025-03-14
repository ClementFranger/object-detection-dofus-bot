class FactoryInstance:
    instance = None

    @classmethod
    def factory(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = cls(*args, **kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance
