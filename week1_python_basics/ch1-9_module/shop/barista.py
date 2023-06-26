from coffee.espresso import Espresso
from coffee.americano import Americano


class Barista:
    def __init__(self, name, available_menus):
        self.name = name
        self.available_menus = available_menus

    # 냉장고 객체를 전달 받음
    def receive_order(self, refrigerator):
        print(f"안녕하세요 바리스타 {self.name} 입니다.")
        print(f"주문 가능한 메뉴입니다. {self.available_menus}\n")

        # 주문 받기
        menu = input("어떤 메뉴를 주문하시겠어요?")
        if menu not in self.available_menus:
            print("주문이 불가능한 메뉴입니다.")
            return

        # 세부 옵션 받기
        coffee = None
        if menu == "에스프레소" or menu == "아메리카노":
            bean = input("원두는 무엇으로 하시겠습니까?")
            shot = int(input("샷은 몇잔으로 하시겠습니까?"))
            if not refrigerator.check_beans(bean, shot):
                print("원두 재고가 부족합니다.")
                return
            if menu == "에스프레소":
                coffee = Espresso(bean, shot)
            else:
                is_hot = bool(int(input("뜨겁게 하시겠습니까? (1: hot, 0: ice")))
                size = input("사이즈는 어떻게 하시겠습니까? (regular, grande, venti)")
                coffee = Americano(bean, shot, is_hot, size)
        return coffee
