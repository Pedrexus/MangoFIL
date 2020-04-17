class Registry:

    registry = {}

    def __init_subclass__(cls, *args, **kwargs):
        Registry.registry[cls.__name__] = cls
        super().__init_subclass__(*args, **kwargs)

    @classmethod
    def get(cls, name: str):
        return cls.registry[name]