documents
: 자비스 팀의 프로젝트 진행 과정을 순서대로 문서화 한 내용입니다, 최종 모델에 적용된 기법들에 대한 근거와 분석들을 보고서 형태로 정리했습니다.
- 1. 소형 객체 탐지 성능 향상 방안 : YOLO 모델의 취약점인 소형 객체에 대한 탐지 성능의 향상을 대주제로 하여 작성된 보고서
- 2. 모듈별 양자화 보존율 : 모델을 양자화 시켰을 때, GPU 기준에서의 성능을 최대한 재현하는것을 대주제로 하여 작성된 보고서
- exp_list.csv : 프로젝트를 진행하면서 실험한 모델들 이름과 스펙을 기록한 Dataframe (FPS가 0인 경우는 NPU에 올라가지 못한 모델)

Model_files
: 최종 제출 모델 관련 파일들입니다.
- nextchip_jvs.yaml : 학습에 사용된 데이터셋 yaml 파일
- v8s_P2_2211_gc-c3g.yaml : 최종 모델을 구조를 선언한 yaml 파일
- v8s_P2_2211_gc-c3g_140epoch.pt : 모델을 구글 코랩 A100을 사용해 150epochs 학습 시킨후 추출한 140epoch의 가중치 파일
- v8s_P2_2211_gc-c3g_140epoch.onnx : pt파일을 변환한 onnx 파일
- v8s_P2_2211_gc-c3g_140epoch.aiwbin : 최종 테스트에 사용된 바이너리 파일

Model_files/nnef
- nnef 및 stats 파일

Model_files/result
- v8s_P2_2211_gc-c3g_140epoch.tar : 위의 바이너리 파일을 통해 test video로 실행한 결과
- v8s_P2_2211_gc-c3g_140epoch.txt : evaluate 결과

성능 재현 가이드
: 최종 제출 모델의 성능을 재현하기 위한 가이드, 내부의 성능 재현 가이드.docx에 서술된 순서대로 진행 

