import torch
from collections import defaultdict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class dLLMCache(metaclass=Singleton):
    gen_interval_steps: int
    prompt_interval_steps: int
    cfg_interval_steps: int
    prompt_length: int
    gen_length: int
    expect_length:int
    transfer_ratio: float
    __cache: defaultdict
    __step_counter: defaultdict

    @classmethod
    def new_instance(
        cls,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        transfer_ratio: float = 0.0,
    ) -> "dLLMCache":
        ins = cls()
        setattr(ins, "prompt_interval_steps", prompt_interval_steps)
        setattr(ins, "gen_interval_steps", gen_interval_steps)
        setattr(ins, "cfg_interval_steps", cfg_interval_steps)
        setattr(ins, "transfer_ratio", transfer_ratio)
        ins.init()
        return ins

    def init(self) -> None:
        self.__cache = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        self.__step_counter = defaultdict(lambda: defaultdict(lambda: 0))

    def reset_cache(self, prompt_length: int = 0,gen_length:int = 0) -> None:
        self.init()
        torch.cuda.empty_cache()
        self.prompt_length = prompt_length
        self.gen_length = gen_length
        self.cache_type = "no_cfg"

    def set_cache(
        self, layer_id: int, feature_name: str, features: torch.Tensor, cache_type: str
    ) -> None:
        if hasattr(self, "expect_length"):
            if cache_type=="gen" and self.gen_length!=self.expect_length:
                tmp=self.__cache[self.cache_type][cache_type][layer_id]["attn"][0]
                b,t,d=tmp.shape
                index=torch.arange(self.expect_length, device=tmp.device).unsqueeze(0).unsqueeze(-1).expand(b, -1, d)
                if feature_name=="kv_cache":
                    self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["k"].scatter_(dim=1,index=index,src=features["k"].clone())
                    self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["v"].scatter_(dim=1,index=index,src=features["v"].clone())
                else:
                    self.__cache[self.cache_type][cache_type][layer_id][feature_name][0].scatter_(dim=1,index=index,src=features.clone())
            else:
                self.__cache[self.cache_type][cache_type][layer_id][feature_name] =  {0: features}
        else:
            self.__cache[self.cache_type][cache_type][layer_id][feature_name] = {
                0: features
            }
    def get_cache(
        self, layer_id: int, feature_name: str, cache_type: str
    ) -> torch.Tensor:
        if hasattr(self, "expect_length"):
            if cache_type=="gen" and self.gen_length!=self.expect_length:
                if feature_name=="kv_cache":
                    output ={
                        "k": self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["k"].clone()[:,:self.expect_length,:],
                        "v": self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]["v"].clone()[:,:self.expect_length,:]
                    }
                else:
                    output = self.__cache[self.cache_type][cache_type][layer_id][feature_name][0][:,:self.expect_length,:]
            else:
                output = self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]
            return output 
        else:
            output = self.__cache[self.cache_type][cache_type][layer_id][feature_name][0]
            return output

    def update_step(self, layer_id: int) -> None:
        self.__step_counter[self.cache_type][layer_id] += 1

    def refresh_gen(self, layer_id: int = 0) -> bool:
        return (self.current_step - 1) % self.gen_interval_steps == 0

    def refresh_prompt(self, layer_id: int = 0) -> bool:
        return (self.current_step - 1) % self.prompt_interval_steps == 0

    def refresh_cfg(self, layer_id: int = 0) -> bool:
        return (
            self.current_step - 1
        ) % self.cfg_interval_steps == 0 or self.current_step <= 5

    @property
    def current_step(self) -> int:
        return max(list(self.__step_counter[self.cache_type].values()), default=1)

    def __repr__(self):
        return f"USE dLLMCache"
