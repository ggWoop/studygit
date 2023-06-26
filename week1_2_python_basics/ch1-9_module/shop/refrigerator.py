class Refrigerator:
    def __init__(self):
        self.beans_inventory = {}

    def add_beans(self, bean, cnt):
        self.beans_inventory[bean] = cnt

    def consume_beans(self, bean, cnt):
        self.beans_inventory[bean] -= cnt

    # 특정 수량만큼 원두 재고가 있는지 체크하는 메서드
    def check_beans(self, bean, cnt):
        if bean not in self.beans_inventory:
            return False
        if self.beans_inventory[bean] < cnt:
            return False
        return True

    def print_inventory(self):
        print("현재 냉장고 상태입니다.")
        if not self.beans_inventory:
            print("원두: 비어있음")
        else:
            print("원두 재고")
            for bean_name, bean_cnt in self.beans_inventory.items():
                print(f"{bean_name}: {bean_cnt}")
